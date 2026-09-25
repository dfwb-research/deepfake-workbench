"""Training runs: :func:`run_experiment` trains a ``dfwb.train/1`` config once per seed, each run
into its own self-describing run directory (see :mod:`dfwb.train.rundir`); :func:`resume_run`
carries an interrupted run on to its end.

Everything that can be checked from the config alone is checked before any data is read and
before anything is written: every component against the installed plugins, the train transforms,
the seeds, the Lightning passthrough, the monitor (against the validation sources' names, which
come from their protocol references alone), the optimiser and its schedule against the built
detector, and the trainer's own options.

A new run claims its directory's name by creating it, before the data is joined; if anything
fails before the run's records are written, the name is given back (the directories removed).

Each run then writes, in order: ``config.resolved.yaml`` and ``fingerprint.txt`` (exactly what
``dfwb config show`` prints for the same config), ``env.json``, ``data.json`` and the ``latest``
link; during training its checkpoints, logs, validation score files and ``resume/`` state; and
once training finishes ``metrics.json`` and ``report.md`` (and ``resume/`` is removed).

A run without validation sources has nothing to monitor: it warns, and keeps ``checkpoints/best``
as a copy of ``checkpoints/last``, so ``run:<dir>`` (which means ``#best``) still resolves.
"""

from __future__ import annotations

import csv
import dataclasses
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
from dfwb.core.config.schema import ClipSection, TrainConfig, check_components
from dfwb.core.errors import ConfigError, ContractError, did_you_mean
from dfwb.core.paths import absolute, require_root, resolve_roots
from dfwb.core.plugins import REGISTRY_NAMES, api, get_registry
from dfwb.core.records.local import ProcessingProfile
from dfwb.core.runmeta import collect_run_info
from dfwb.core.seed import seed_everything
from dfwb.data.transforms import build_transforms
from dfwb.eval.aggregate import check_eval_config
from dfwb.models.detector import AssembledDetector, build_detector
from dfwb.train import rundir
from dfwb.train.callbacks import SafetensorsCheckpoint, build_callbacks
from dfwb.train.datamodule import ProtocolDataModule, SourceData, source_names
from dfwb.train.loggers import build_loggers
from dfwb.train.module import DetectorModule, check_monitor, check_monitor_syntax
from dfwb.train.optim import build_optimizer
from dfwb.train.resume import ResumeCheckpoint, SavedState, read_state
from dfwb.train.schedules import build_schedule

__all__ = ["RunResult", "device_options", "resolve_precision", "resume_run", "run_experiment"]

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
    shipped_profiles: Sequence[ProcessingProfile] = ()


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
    """Every check the config allows on its own, before a detector is built or any data read."""
    check_components(config, {name: get_registry(name) for name in REGISTRY_NAMES})
    check_eval_config(config.eval.metrics, config.eval.aggregate)
    _check_transforms(config)
    _check_seeds(config.run.seeds)
    _check_lightning(config.train.lightning)
    check_monitor_syntax(config.train.monitor, config.eval.metrics)


def _check_monitor_sources(config: TrainConfig, work_root: Path) -> None:
    """Check the monitor against the validation sources' names, which come from their protocol
    references alone: a typo in a per-source monitor fails before any processed store is
    read."""
    names = source_names(config.data.val, work_root=work_root)
    check_monitor(config.train.monitor, config.eval.metrics, names)


def _check_optimization(config: TrainConfig, detector: AssembledDetector) -> None:
    """Build the optimiser and its schedule once over ``detector`` (then drop them), so a group
    or parameter typo fails before any data is read."""
    optimizer = build_optimizer(config.optim, detector)
    build_schedule(config.schedule, optimizer, steps_per_epoch=1, epochs=config.train.max_epochs)


def _plugin_callbacks(config: TrainConfig) -> list[Callback]:
    """``train.callbacks``, built through the ``callbacks`` registry (their params were already
    checked with the rest of the config).

    Raises:
        ContractError: An entry builds something other than a Lightning callback.
    """
    built: list[Callback] = []
    for index, spec in enumerate(config.train.callbacks):
        callback = api.callbacks.build(spec.name, **spec.params)
        if not isinstance(callback, Callback):
            raise ContractError(
                f"train.callbacks[{index}]: callbacks/{spec.name} built a "
                f"{type(callback).__name__}, not a Lightning callback",
                hint="a callbacks registry target must build a lightning.pytorch Callback",
            )
        built.append(callback)
    return built


# ----------------------------------------------------------------------------- the trainer


def resolve_precision(precision: str, options: Mapping[str, Any]) -> str:
    """The Lightning precision a run trains in: ``precision`` itself, unless it is ``auto``,
    which picks per device -- ``bf16-mixed`` on a CUDA GPU that supports bfloat16, ``16-mixed`` on
    any other CUDA GPU, ``32-true`` everywhere else (the CPU included). ``options`` are the
    trainer options the device comes from (``accelerator``: ``auto`` means a GPU when there is
    one)."""
    if precision != "auto":
        return precision
    accelerator = str(options.get("accelerator", "auto")).lower()
    cuda = accelerator in ("gpu", "cuda") or (accelerator == "auto" and torch.cuda.is_available())
    if not cuda:
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"


def _trainer(plan: _Plan, run_dir: Path, callbacks: list[Callback], loggers: Any) -> L.Trainer:
    train = plan.config.train
    options: dict[str, Any] = {
        "accelerator": "auto",
        "devices": 1,
        "max_epochs": train.max_epochs,
        "enable_checkpointing": False,
        "enable_progress_bar": plan.progress,
        "enable_model_summary": plan.progress,
        "default_root_dir": run_dir,
        "logger": loggers,
        "callbacks": callbacks,
    }
    options.update(train.lightning)
    if not plan.progress:
        # both print to stdout, which then holds nothing but the results (--json), whatever
        # train.lightning asks for
        options.update(enable_progress_bar=False, enable_model_summary=False)
    if plan.device is not None:
        options.update(device_options(plan.device))
    options["precision"] = resolve_precision(train.precision, options)
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
        # what train.precision resolved to on this device (``auto`` picks per device)
        "precision": str(trainer.precision),
        "git": git,
        "plugins": info.plugins,
    }


def _write_run_dir(
    plan: _Plan, run_dir: Path, seed: int, trainer: L.Trainer, data: Mapping[str, Any]
) -> None:
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
            # counted from 1, like "epochs" (the trainer's own epoch index starts at 0)
            "best_epoch": None if best["best_epoch"] is None else best["best_epoch"] + 1,
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
        f"- Trained: {metrics['epochs']} epoch(s), {metrics['global_step']} optimiser step(s); "
        "epochs are counted from 1",
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
            f"{_number(monitor['best'])} after epoch {monitor['best_epoch']} of "
            f"{metrics['epochs']}"
        )
        if monitor["fallback"]:
            lines.append(
                f"- The monitor fell back to `{monitor['key']}` ({monitor['mode']}): "
                f"`{config.train.monitor}` was undefined on every validation source"
            )
    undefined = [
        f"`{metric}` on `{source['name']}`"
        for source in data["val"]
        for metric in config.eval.metrics
        if f"val/{source['name']}/{metric}" not in metrics["val"]
    ]
    if metrics["val"] and undefined:
        lines.append(
            "- Undefined in the last validation, so left out of the `val/video_<metric>` means: "
            + ", ".join(undefined)
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


def _prepare(
    plan: _Plan,
    seed: int,
    run_dir: Path,
    detector: AssembledDetector,
    restore: SavedState | None,
) -> tuple[L.Trainer, ProtocolDataModule, dict[str, Any]]:
    """Build the trainer, join the data, then write the new run's records into its (reserved)
    directory -- or, for a resumed run, add the resume's own records."""
    config = plan.config
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
        *_plugin_callbacks(config),
        # last, so the state it saves is every other callback's once the epoch is over
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
        shipped_profiles=plan.shipped_profiles,
    )
    datamodule.setup("fit")
    detector.meta = dataclasses.replace(
        detector.meta, training_data=_training_data(datamodule.train_sources)
    )
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

    return trainer, datamodule, data


# How a clip sampling mode is recorded in a detector's input spec: a random window is still a run
# of consecutive frames, just placed at random.
_SAMPLING_INPUT = {
    "uniform": "uniform",
    "consecutive": "consecutive",
    "random-window": "consecutive",
}


def _input_overrides(config: TrainConfig) -> dict[str, Any]:
    """The detector's input spec overrides: the clip regime it trains and validates on
    (``data.clip``'s frames and sampling), so scoring it later builds the same clips, then
    whatever ``model.input`` sets, which wins."""
    clip: ClipSection = config.data.clip
    overrides: dict[str, Any] = {"frames": clip.frames, "sampling": _SAMPLING_INPUT[clip.sampling]}
    if config.model.input is not None:
        overrides.update(config.model.input.overrides())
    return overrides


def _training_data(sources: Sequence[SourceData]) -> tuple[str, ...]:
    """One pinned protocol reference per training protocol, ``<pack>:<dataset>/<scheme>@<pack
    version>``, in config order, each once (the split and ``where`` filters are in
    ``data.json``)."""
    refs = [
        f"{source.protocol.pack.name}:{source.protocol.ref}@{source.protocol.pack_version}"
        for source in sources
    ]
    return tuple(dict.fromkeys(refs))


def _fit(plan: _Plan, seed: int, *, run_dir: Path | None, restore: SavedState | None) -> RunResult:
    config = plan.config
    seed_everything(seed)
    detector = build_detector(
        config.model,
        input_spec_overrides=_input_overrides(config),
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
    reservation: rundir.Reservation | None = None
    if run_dir is None:
        reservation = rundir.reserve_run_dir(plan.runs_root / config.run.name, _timestamp(), seed)
        run_dir = reservation.path
    try:
        trainer, datamodule, data = _prepare(plan, seed, run_dir, detector, restore)
    except BaseException:
        # nothing of the run was written yet: give its name back
        if reservation is not None:
            reservation.release()
        raise

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
    shipped_profiles: Sequence[ProcessingProfile] = (),
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
        shipped_profiles: The processing profiles the framework ships: when a store cannot
            serve the detector's input, the refusal lists those that would.

    Raises:
        ConfigError: A problem with the config, found before any data is read; or with the data,
            found before any training.
    """
    train_config = _train_config(config)
    _check_config(train_config)
    if device is not None:
        device_options(device)
    work = _work_root(work_root)
    _check_monitor_sources(train_config, work)
    if train_config.data.test:
        _log.warning(
            "data.test is not run by `dfwb train` in this version; score the trained run on "
            "it with `dfwb score` and evaluate that with `dfwb eval`"
        )
    if train_config.run.output_root:
        root = absolute(train_config.run.output_root)
    else:
        root = Path(runs_root) if runs_root is not None else require_root("runs", resolve_roots())
    plan = _Plan(config, train_config, work, root, device, progress, argv, shipped_profiles)
    return [_fit(plan, seed, run_dir=None, restore=None) for seed in train_config.run.seeds]


def resume_run(
    run_dir: Path,
    *,
    config: LoadedConfig | None = None,
    work_root: Path | None = None,
    device: str | None = None,
    progress: bool = True,
    argv: Sequence[str] | None = None,
    shipped_profiles: Sequence[ProcessingProfile] = (),
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
            hint="a finished run cannot be extended: more epochs is a new experiment "
            "(dfwb train -c <config>)",
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
    work = _work_root(work_root)
    _check_monitor_sources(train_config, work)
    plan = _Plan(
        loaded, train_config, work, run_dir.parent.parent, device, progress, argv, shipped_profiles
    )
    return _fit(plan, int(saved.payload["seed"]), run_dir=run_dir, restore=saved)
