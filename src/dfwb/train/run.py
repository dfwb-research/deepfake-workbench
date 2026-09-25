"""Training runs: :func:`run_experiment` trains a ``dfwb.train/1`` config once per seed, each run
into its own self-describing run directory (see :mod:`dfwb.train.rundir`); :func:`resume_run`
carries an interrupted run on to its end.

Everything that can be checked from the config alone is checked before any data is read and
before anything is written: every component against the installed plugins, the train transforms,
the optimiser and its schedule against the built detector, the monitor, the seeds, and the
trainer's own options. The monitor is checked once more against the validation sources' names as
soon as the data is joined -- still before any training, and before the run directory exists.

Each run then writes, in order: ``config.resolved.yaml`` and ``fingerprint.txt`` (exactly what
``dfwb config show`` prints for the same config), ``env.json``, ``data.json`` and the ``latest``
link; during training its checkpoints, logs, validation score files and ``resume/`` state; and
once training finishes ``metrics.json`` and ``report.md`` (and ``resume/`` is removed).

A run without validation sources has nothing to monitor: it warns, and keeps ``checkpoints/best``
as a copy of ``checkpoints/last``, so ``run:<dir>`` (which means ``#best``) still resolves.
"""

from __future__ import annotations

import csv
import inspect
import json
import logging
import math
import shutil
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as L
import torch
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.pytorch.callbacks import Callback

from dfwb.core.config.loader import LoadedConfig, dump_yaml, load_config
from dfwb.core.config.schema import TrainConfig, check_components
from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.paths import absolute, require_root, resolve_roots
from dfwb.core.plugins import REGISTRY_NAMES, get_registry
from dfwb.core.runmeta import collect_run_info
from dfwb.core.seed import seed_everything
from dfwb.data.transforms import build_transforms
from dfwb.models.detector import AssembledDetector, build_detector
from dfwb.train import rundir
from dfwb.train.callbacks import SafetensorsCheckpoint, build_callbacks
from dfwb.train.datamodule import ProtocolDataModule, SourceData
from dfwb.train.loggers import build_loggers
from dfwb.train.module import DetectorModule
from dfwb.train.optim import build_optimizer
from dfwb.train.resume import ResumeCheckpoint, SavedState, read_state
from dfwb.train.schedules import build_schedule

__all__ = ["RunResult", "device_options", "resume_run", "run_experiment"]

_log = logging.getLogger(__name__)

_DEVICES = ("cpu", "cuda", "gpu")
_CSV_LOG = "metrics.csv"
_CSV_STASH = ".metrics.before-resume.csv"
_TRAIN_EXTRA = 'pip install "deepfake-workbench[train]"'


@dataclass(frozen=True)
class RunResult:
    """One finished run: its directory, its seed, and what ``metrics.json`` holds."""

    run_dir: Path
    seed: int
    metrics: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"run_dir": str(self.run_dir), "seed": self.seed, "metrics": self.metrics}


@dataclass(frozen=True)
class _Plan:
    loaded: LoadedConfig
    config: TrainConfig
    work_root: Path
    runs_root: Path
    device: str | None
    progress: bool
    argv: Sequence[str] | None


# ------------------------------------------------------------------------------ checks


def device_options(device: str) -> dict[str, Any]:
    """Lightning ``Trainer`` options for ``--device``: ``cpu``, ``cuda`` (or ``gpu``), or
    ``cuda:<index>`` for one particular GPU. Always one device: training runs in one process.

    Raises:
        ConfigError: ``device`` is none of those.
    """
    name, sep, index = device.partition(":")
    if name == "cpu" and not sep:
        return {"accelerator": "cpu", "devices": 1}
    if name in ("cuda", "gpu"):
        if not sep:
            return {"accelerator": "gpu", "devices": 1}
        if index.isdigit():
            return {"accelerator": "gpu", "devices": [int(index)]}
    raise ConfigError(
        f"--device: {device!r} is not a device{did_you_mean(name, _DEVICES)}",
        hint="use cpu, cuda, or cuda:<index> for one particular GPU",
    )


def _train_config(loaded: LoadedConfig) -> TrainConfig:
    if not isinstance(loaded.model, TrainConfig):
        raise ConfigError(
            f"training needs a dfwb.train/1 config, not {loaded.schema}",
            hint="set 'schema: dfwb.train/1' (dfwb config init writes a starter file)",
        )
    return loaded.model


def _check_transforms(config: TrainConfig) -> None:
    """Build the train transforms once, so one that can never work (``normalize``, which input
    adaptation owns) fails here rather than after the data is joined."""
    if not config.data.transforms.train:
        return
    try:
        build_transforms(config.data.transforms.train)
    except ConfigError as exc:
        lines = [
            f"data.{line}" if line.startswith("transforms.") else line
            for line in exc.message.splitlines()
        ]
        raise ConfigError("\n".join(lines), hint=exc.hint) from None


def _check_seeds(seeds: Sequence[int]) -> None:
    seen: set[int] = set()
    for seed in seeds:
        if seed in seen:
            raise ConfigError(
                f"run.seeds: {seed} appears more than once",
                hint="list each seed once: every seed is one run",
            )
        seen.add(seed)


def _check_lightning(options: Mapping[str, Any]) -> None:
    accepted = set(inspect.signature(L.Trainer.__init__).parameters) - {"self"}
    for key in options:
        if key not in accepted:
            raise ConfigError(
                f"train.lightning: {key!r} is not a Lightning Trainer argument"
                f"{did_you_mean(key, accepted)}",
                hint="train.lightning passes its keys to lightning.pytorch.Trainer",
            )


def _check_config(config: TrainConfig) -> None:
    """Every check the config allows before a detector is built or any data read."""
    check_components(config, {name: get_registry(name) for name in REGISTRY_NAMES})
    _check_transforms(config)
    _check_seeds(config.run.seeds)
    _check_lightning(config.train.lightning)


def _check_optimization(config: TrainConfig, detector: AssembledDetector) -> None:
    """Build the optimiser and its schedule once over ``detector`` (then drop them), so a group
    or parameter typo fails before any data is read."""
    optimizer = build_optimizer(config.optim, detector)
    build_schedule(config.schedule, optimizer, steps_per_epoch=1, epochs=config.train.max_epochs)


# ----------------------------------------------------------------------------- the trainer


def _trainer(plan: _Plan, run_dir: Path, callbacks: list[Callback], loggers: Any) -> L.Trainer:
    train = plan.config.train
    options: dict[str, Any] = {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": train.max_epochs,
        "precision": train.precision,
        "enable_checkpointing": False,
        # both print to stdout, which must hold nothing but the results in JSON mode
        "enable_progress_bar": plan.progress,
        "enable_model_summary": plan.progress,
        "default_root_dir": run_dir,
        "logger": loggers,
        "callbacks": callbacks,
    }
    options.update(train.lightning)
    if plan.device is not None:
        options.update(device_options(plan.device))
    try:
        return L.Trainer(**options)
    except (MisconfigurationException, ValueError, TypeError) as exc:
        raise ConfigError(
            f"train: Lightning refused these trainer options: {exc}",
            hint="check train.precision, train.lightning and --device",
        ) from None


def _device_name(trainer: L.Trainer) -> str:
    device = trainer.strategy.root_device
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


# ------------------------------------------------------------------------- what a run records


def _source_record(source: SourceData) -> dict[str, Any]:
    summary = source.index.summary()
    protocol = summary["sources"][0]["protocol"]
    return {
        "name": source.name,
        "protocol": {
            "ref": source.entry.protocol,
            "id": protocol["id"],
            "split": protocol["split"],
            "where": protocol["where"],
            "pack": protocol["pack"],
            "pack_version": protocol["pack_version"],
            "scheme_sha256": protocol["scheme_sha256"],
        },
        "profile": {"id": source.profile.profile_id(), "sha256": source.profile.sha256()},
        "input_adaptation": {
            "derived_crop": source.adaptation.derived_crop,
            "mismatch": source.adaptation.mismatch,
            "reason": source.adaptation.reason,
        },
        "index": summary,
    }


def _data_record(config: TrainConfig, datamodule: ProtocolDataModule) -> dict[str, Any]:
    data = config.data
    return {
        "processing": data.processing,
        "labels": data.labels,
        "clip": data.clip.model_dump(mode="json"),
        "pairs": data.pairs,
        "allow_input_mismatch": data.allow_input_mismatch,
        "train": [_source_record(source) for source in datamodule.train_sources],
        "val": [_source_record(source) for source in datamodule.val_sources],
    }


def _env_record(plan: _Plan, seed: int, trainer: L.Trainer) -> dict[str, Any]:
    info = collect_run_info(plan.argv)
    env = {**info.env, "device": _device_name(trainer)}
    git = None if info.git is None else {"commit": info.git.commit, "dirty": info.git.dirty}
    return {
        "seed": seed,
        "created": info.created,
        "command": info.command,
        "env": env,
        "git": git,
        "plugins": info.plugins,
    }


def _write_run_dir(
    plan: _Plan, run_dir: Path, seed: int, trainer: L.Trainer, data: Mapping[str, Any]
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / rundir.CONFIG_FILE).write_text(dump_yaml(plan.loaded.data), encoding="utf-8")
    (run_dir / rundir.FINGERPRINT_FILE).write_text(f"{plan.loaded.fingerprint}\n", "utf-8")
    rundir.write_json(run_dir / rundir.ENV_FILE, _env_record(plan, seed, trainer))
    rundir.write_json(run_dir / rundir.DATA_FILE, data)
    rundir.point_latest(run_dir)


def _record_resume(
    plan: _Plan, run_dir: Path, trainer: L.Trainer, data: Mapping[str, Any], epoch: int
) -> None:
    """Before a resumed fit: add where and how it resumed to ``env.json`` (the machine, versions
    or command may differ from the start's), warn if the joined data changed, and set the
    finished epochs' CSV log rows aside."""
    env = rundir.read_json(run_dir / rundir.ENV_FILE)
    resumed = {"epoch": epoch, **_env_record(plan, int(env["seed"]), trainer)}
    del resumed["seed"]
    env.setdefault("resumed", []).append(resumed)
    rundir.write_json(run_dir / rundir.ENV_FILE, env)
    if json.loads(json.dumps(data)) != rundir.read_json(run_dir / rundir.DATA_FILE):
        _log.warning(
            "%s: the joined data differs from what the run started with (see %s); the "
            "remaining epochs train on the data as it is now",
            run_dir,
            rundir.DATA_FILE,
        )
    _stash_csv_log(run_dir, epoch)


def _finite(value: float | None) -> float | None:
    """``value``, or ``None`` for a value JSON cannot hold (NaN, infinity)."""
    return value if value is not None and math.isfinite(value) else None


def _metrics_record(
    plan: _Plan, seed: int, trainer: L.Trainer, module: DetectorModule
) -> dict[str, Any]:
    monitor: dict[str, Any] | None = None
    if plan.config.data.val:
        callbacks = getattr(trainer, "callbacks", [])
        (saver,) = [cb for cb in callbacks if isinstance(cb, SafetensorsCheckpoint)]
        best = saver.state_dict()
        monitor = {
            "key": module.monitor.key,
            "mode": module.monitor.mode,
            "fallback": module.monitor.fallback,
            "best": _finite(best["best_value"]),
            "best_epoch": best["best_epoch"],
        }
    return {
        "seed": seed,
        "fingerprint": plan.loaded.fingerprint,
        "epochs": trainer.current_epoch,
        "global_step": trainer.global_step,
        "monitor": monitor,
        "val": {key: _finite(value) for key, value in module.val_metrics.items()},
    }


def _number(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.4f}"


def _report(plan: _Plan, run_dir: Path, metrics: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    config = plan.config
    lines = [
        f"# {config.run.name}, seed {metrics['seed']}",
        "",
        f"- Run: `{run_dir.parent.name}/{run_dir.name}`",
        f"- Config fingerprint: `{metrics['fingerprint']}`",
        f"- Trained: {metrics['epochs']} epoch(s), {metrics['global_step']} optimiser step(s)",
    ]
    monitor = metrics["monitor"]
    if monitor is None:
        lines.append(
            "- No validation sources: monitoring was off, and `checkpoints/best` is a copy of "
            "`checkpoints/last`"
        )
    else:
        lines.append(
            f"- Best checkpoint (`checkpoints/best`): `{monitor['key']}` ({monitor['mode']}) = "
            f"{_number(monitor['best'])} after epoch {monitor['best_epoch']}"
        )
    lines += [
        "",
        "## Data",
        "",
        "| role | source | protocol | split | videos used | in split | excluded |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for role in ("train", "val"):
        for source in data[role]:
            counts = source["index"]["sources"][0]["counts"]
            excluded = ", ".join(f"{k}: {v}" for k, v in counts["excluded"].items()) or "none"
            lines.append(
                f"| {role} | {source['name']} | `{source['protocol']['ref']}` | "
                f"{source['protocol']['split']} | {counts['included']} | {counts['in_split']} | "
                f"{excluded} |"
            )
    if metrics["val"]:
        lines += ["", "## Validation, last epoch", "", "| value | |", "|---|---:|"]
        lines += [f"| `{key}` | {_number(value)} |" for key, value in metrics["val"].items()]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- resumed CSV logs


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    keys = sorted({key for row in rows for key in row})
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _finished_epoch(row: Mapping[str, str], epochs_done: int) -> bool:
    epoch = row.get("epoch")
    return not epoch or int(float(epoch)) < epochs_done


def _stash_csv_log(run_dir: Path, epochs_done: int) -> None:
    """Before a resumed fit: set the rows logged by the finished epochs aside (Lightning's CSV
    logger starts its file afresh), dropping those of the epoch that was cut short."""
    logs = run_dir / rundir.LOGS_DIR
    rows = _read_rows(logs / _CSV_STASH) + _read_rows(logs / _CSV_LOG)
    if not rows:
        return
    _write_rows(logs / _CSV_STASH, [row for row in rows if _finished_epoch(row, epochs_done)])
    (logs / _CSV_LOG).unlink(missing_ok=True)


def _merge_csv_log(run_dir: Path) -> None:
    """After a resumed fit (finished or not): the set-aside rows, then the new ones."""
    logs = run_dir / rundir.LOGS_DIR
    stash = logs / _CSV_STASH
    if not stash.is_file():
        return
    _write_rows(logs / _CSV_LOG, _read_rows(stash) + _read_rows(logs / _CSV_LOG))
    stash.unlink()


# ---------------------------------------------------------------------------------- one run


def _timestamp() -> str:
    return rundir.run_stamp()


def _fit(plan: _Plan, seed: int, *, run_dir: Path | None, restore: SavedState | None) -> RunResult:
    config = plan.config
    seed_everything(seed)
    detector = build_detector(
        config.model,
        input_spec_overrides=config.model.input.overrides() if config.model.input else None,
        source=f"run:{plan.loaded.fingerprint}",
    )
    _check_optimization(config, detector)
    module = DetectorModule(
        detector,
        loss=config.loss,
        optim=config.optim,
        schedule=config.schedule,
        train=config.train,
        eval=config.eval,
    )
    if run_dir is None:
        parent = plan.runs_root / config.run.name
        run_dir = parent / rundir.reserve_name(parent, _timestamp(), seed)
    monitored = bool(config.data.val)
    early_stop = config.train.early_stop
    callbacks = [
        *build_callbacks(
            run_dir,
            config.model,
            early_stop_patience=early_stop.patience if early_stop else None,
            early_stop_min_delta=early_stop.min_delta if early_stop else 0.0,
            nan_tolerance=config.train.nan_tolerance,
            heartbeat_every=config.train.heartbeat_steps,
            best_is_last=not monitored,
        ),
        ResumeCheckpoint(run_dir / rundir.RESUME_DIR, seed=seed, restore=restore),
    ]
    loggers = build_loggers(
        run_dir,
        tensorboard="tensorboard" in config.train.loggers,
        wandb={} if "wandb" in config.train.loggers else None,
    )
    trainer = _trainer(plan, run_dir, callbacks, loggers)

    datamodule = ProtocolDataModule(
        config.data,
        input_spec=detector.meta.input,
        work_root=plan.work_root,
        seed=seed,
        pairs=config.data.pairs,
        allow_mismatch=config.data.allow_input_mismatch,
    )
    datamodule.setup("fit")
    module.check_monitor([source.name for source in datamodule.val_sources])
    if not monitored:
        _log.warning(
            "%s: no validation sources, so monitoring is off: checkpoints/best is kept as a "
            "copy of checkpoints/last",
            config.run.name,
        )
    data = _data_record(config, datamodule)
    if restore is None:
        _write_run_dir(plan, run_dir, seed, trainer, data)
    else:
        _record_resume(plan, run_dir, trainer, data, int(restore.payload["epoch"]))

    try:
        with warnings.catch_warnings():
            # a resumed run's logs/ is not empty, and the CSV rows it holds are carried over
            warnings.filterwarnings("ignore", message="Experiment logs directory .* not empty")
            trainer.fit(module, datamodule=datamodule)
    except SystemExit:
        _log.warning("%s: interrupted; carry on with: dfwb train --resume %s", run_dir, run_dir)
        raise
    finally:
        _merge_csv_log(run_dir)

    metrics = _metrics_record(plan, seed, trainer, module)
    rundir.write_json(run_dir / rundir.METRICS_FILE, metrics)
    (run_dir / rundir.REPORT_FILE).write_text(_report(plan, run_dir, metrics, data), "utf-8")
    shutil.rmtree(run_dir / rundir.RESUME_DIR, ignore_errors=True)
    return RunResult(run_dir, seed, metrics)


# ------------------------------------------------------------------------------- the API


def _work_root(work_root: Path | None) -> Path:
    return Path(work_root) if work_root is not None else require_root("work", resolve_roots())


def run_experiment(
    config: LoadedConfig,
    *,
    work_root: Path | None = None,
    runs_root: Path | None = None,
    device: str | None = None,
    progress: bool = True,
    argv: Sequence[str] | None = None,
) -> list[RunResult]:
    """Train ``config`` once per seed of ``run.seeds``, in order; one run directory each.

    Args:
        config: A loaded ``dfwb.train/1`` config (:func:`dfwb.core.config.load_config`).
        work_root: Where the processed stores live (default: the work root).
        runs_root: Where runs go when the config sets no ``run.output_root`` (default: the runs
            root, which is ``./runs`` unless configured).
        device: ``cpu``, ``cuda`` or ``cuda:<index>``; default: the config's
            ``train.lightning.accelerator``, else any GPU, else the CPU.
        progress: Show Lightning's progress bar and model summary (both print to stdout).
        argv: The command line to record in ``env.json`` (default: this process's).

    Raises:
        ConfigError: A problem with the config, found before any data is read; or with the data,
            found before any training.
    """
    train_config = _train_config(config)
    _check_config(train_config)
    if device is not None:
        device_options(device)
    work = _work_root(work_root)
    if train_config.run.output_root:
        root = absolute(train_config.run.output_root)
    else:
        root = Path(runs_root) if runs_root is not None else require_root("runs", resolve_roots())
    plan = _Plan(config, train_config, work, root, device, progress, argv)
    return [_fit(plan, seed, run_dir=None, restore=None) for seed in train_config.run.seeds]


def resume_run(
    run_dir: Path,
    *,
    config: LoadedConfig | None = None,
    work_root: Path | None = None,
    device: str | None = None,
    progress: bool = True,
    argv: Sequence[str] | None = None,
) -> RunResult:
    """Carry the interrupted run in ``run_dir`` on from the end of its last finished epoch.

    It trains the run's own ``config.resolved.yaml``; ``config``, when given, must be the same
    experiment (the same fingerprint).

    Raises:
        ConfigError: ``run_dir`` is not a run directory, has nothing to resume, or ``config`` is
            a different experiment.
        ContractError: The run's resolved config no longer matches its fingerprint.
    """
    run_dir = absolute(run_dir)
    if not rundir.is_run_dir(run_dir):
        raise ConfigError(
            f"{run_dir} is not a run directory (it has no {rundir.FINGERPRINT_FILE} and "
            f"{rundir.CONFIG_FILE})",
            hint="pass a directory made by `dfwb train`; `dfwb runs list` shows them",
        )
    saved = read_state(run_dir / rundir.RESUME_DIR)
    if saved is None:
        why = (
            "the run finished"
            if (run_dir / rundir.METRICS_FILE).is_file()
            else "it stopped before its first epoch ended"
        )
        raise ConfigError(
            f"{run_dir}: nothing to resume ({why})",
            hint="start a new run with dfwb train -c <config>",
        )
    recorded = (run_dir / rundir.FINGERPRINT_FILE).read_text("utf-8").strip()
    loaded = load_config(run_dir / rundir.CONFIG_FILE)
    if loaded.fingerprint != recorded:
        raise ContractError(
            f"{run_dir}: {rundir.CONFIG_FILE} no longer matches {rundir.FINGERPRINT_FILE} "
            f"(fingerprint {loaded.fingerprint[:12]}, recorded {recorded[:12]})",
            hint="the resolved config was edited: put it back, or start a new run",
        )
    if config is not None and config.fingerprint != recorded:
        raise ConfigError(
            f"{config.sources[-1]} is not the config of {run_dir} (fingerprint "
            f"{config.fingerprint[:12]}, the run's {recorded[:12]})",
            hint="resume without -c: a run always carries on with its own resolved config",
        )
    train_config = _train_config(loaded)
    _check_config(train_config)
    if device is not None:
        device_options(device)
    plan = _Plan(
        loaded, train_config, _work_root(work_root), run_dir.parent.parent, device, progress, argv
    )
    return _fit(plan, int(saved.payload["seed"]), run_dir=run_dir, restore=saved)
