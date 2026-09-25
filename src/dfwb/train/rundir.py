"""The run directory: its layout, and reading runs back -- without torch.

One run of one seed lives at ``<runs root>/<run.name>/<YYYYmmdd-HHMMSS>-s<seed>/``::

    config.resolved.yaml    the resolved config (what ``dfwb config show`` prints)
    fingerprint.txt         its fingerprint, the same for every run of the same experiment
    env.json                seed, versions, device, git state, command, plugin providers
    data.json               per source: protocol, pack version, split hash, profile, join counts
    checkpoints/{best,last}/{model.safetensors,detector.json}
    logs/                   metrics.csv (always), TensorBoard, heartbeat.json
    scores/val/<source>.scores.{csv,meta.json}
    report.md  metrics.json written when training finishes

``<runs root>/<run.name>/latest`` is a relative symlink to the newest run of that name. While a run
is training, or after it was interrupted, ``resume/`` holds what ``dfwb train --resume`` needs;
it is removed once the run finishes.

Everything here is plain files and JSON, so ``dfwb runs`` lists and shows runs on a machine
without torch.
"""

from __future__ import annotations

import datetime
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from dfwb.core.errors import ConfigError, did_you_mean
from dfwb.core.paths import absolute

__all__ = [
    "CHECKPOINTS_DIR",
    "CONFIG_FILE",
    "DATA_FILE",
    "ENV_FILE",
    "FINGERPRINT_FILE",
    "LATEST",
    "LOGS_DIR",
    "METRICS_FILE",
    "REPORT_FILE",
    "RESUME_DIR",
    "RESUME_STATE_FILE",
    "RunSummary",
    "data_rows",
    "find_run",
    "is_run_dir",
    "list_runs",
    "point_latest",
    "read_json",
    "read_run",
    "reserve_name",
    "run_stamp",
    "write_json",
]

CONFIG_FILE = "config.resolved.yaml"
FINGERPRINT_FILE = "fingerprint.txt"
ENV_FILE = "env.json"
DATA_FILE = "data.json"
METRICS_FILE = "metrics.json"
REPORT_FILE = "report.md"
CHECKPOINTS_DIR = "checkpoints"
LOGS_DIR = "logs"
RESUME_DIR = "resume"
RESUME_STATE_FILE = "state.json"
LATEST = "latest"

Status = Literal["completed", "incomplete"]


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON, whole or not at all (written beside, then renamed)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def reserve_name(parent: Path, stamp: str, seed: int) -> str:
    """``<stamp>-s<seed>``, or with ``-2``, ``-3``, ... appended when a run of the same seed
    already started in the same second."""
    base = f"{stamp}-s{seed}"
    name, n = base, 1
    while (parent / name).exists() or (parent / name).is_symlink():
        n += 1
        name = f"{base}-{n}"
    return name


def point_latest(run_dir: Path) -> None:
    """Point ``<run dir>/../latest`` at ``run_dir``, relatively (so the runs root can move)."""
    link = run_dir.parent / LATEST
    tmp = run_dir.parent / f".{LATEST}.tmp-{os.getpid()}"
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(run_dir.name, target_is_directory=True)
    tmp.replace(link)


def is_run_dir(path: Path) -> bool:
    return (path / FINGERPRINT_FILE).is_file() and (path / CONFIG_FILE).is_file()


@dataclass(frozen=True)
class RunSummary:
    """What ``dfwb runs list`` shows of one run."""

    path: Path
    name: str
    seed: int | None
    created: str | None
    status: Status
    epochs: int | None
    fingerprint: str
    monitor: dict[str, Any] | None
    latest: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run": self.path.name,
            "path": str(self.path),
            "seed": self.seed,
            "created": self.created,
            "status": self.status,
            "epochs": self.epochs,
            "fingerprint": self.fingerprint,
            "monitor": self.monitor,
            "latest": self.latest,
        }


def _optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def _latest_target(name_dir: Path) -> Path | None:
    link = name_dir / LATEST
    if not link.is_symlink():
        return None
    return name_dir / link.readlink()


def read_run(path: Path) -> RunSummary:
    """Summarise the run directory at ``path``.

    Raises:
        ConfigError: ``path`` is not a run directory.
    """
    if not is_run_dir(path):
        raise ConfigError(
            f"{path} is not a run directory (it has no {FINGERPRINT_FILE} and {CONFIG_FILE})",
            hint="pass a directory made by `dfwb train`; `dfwb runs list` shows them",
        )
    env = _optional_json(path / ENV_FILE)
    metrics = _optional_json(path / METRICS_FILE)
    resume = _optional_json(path / RESUME_DIR / RESUME_STATE_FILE)
    completed = bool(metrics)
    epochs = metrics.get("epochs") if completed else resume.get("epoch")
    seed = env.get("seed", metrics.get("seed", resume.get("seed")))
    latest = _latest_target(path.parent)
    return RunSummary(
        path=path,
        name=path.parent.name,
        seed=seed,
        created=env.get("created"),
        status="completed" if completed else "incomplete",
        epochs=epochs,
        fingerprint=(path / FINGERPRINT_FILE).read_text("utf-8").strip(),
        monitor=metrics.get("monitor") if completed else None,
        latest=latest is not None and latest.name == path.name,
    )


def list_runs(root: Path) -> list[RunSummary]:
    """Every run under ``root``, by run name and then by directory name (so, by start time)."""
    if not root.is_dir():
        return []
    runs: list[RunSummary] = []
    for name_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        for run_dir in sorted(name_dir.iterdir()):
            if run_dir.is_dir() and not run_dir.is_symlink() and is_run_dir(run_dir):
                runs.append(read_run(run_dir))
    return runs


def _resolve(candidate: Path) -> Path | None:
    """``candidate`` as a run directory: itself, or the ``latest`` run of a run-name
    directory (or of a ``latest`` link)."""
    if candidate.is_symlink() and candidate.name == LATEST:
        target = _latest_target(candidate.parent)
        return target if target is not None and is_run_dir(target) else None
    if is_run_dir(candidate):
        return candidate
    if candidate.is_dir():
        target = _latest_target(candidate)
        if target is not None and is_run_dir(target):
            return target
    return None


def find_run(ref: str, root: Path) -> Path:
    """The run directory ``ref`` names: a path to a run directory, a run name (its ``latest``
    run), ``<name>/latest`` or ``<name>/<run>`` -- as a path first, then under ``root``.

    Raises:
        ConfigError: ``ref`` names no run.
    """
    for candidate in (absolute(ref), root / ref):
        if candidate.exists() or candidate.is_symlink():
            found = _resolve(candidate)
            if found is not None:
                return found
            if candidate.is_dir() and not candidate.is_symlink():
                raise ConfigError(
                    f"{candidate} is not a run directory, and has no {LATEST} run",
                    hint="`dfwb runs list` shows the runs under the runs root",
                )
    known: list[str] = []
    if root.is_dir():
        for name_dir in (p for p in root.iterdir() if p.is_dir()):
            known.append(name_dir.name)
            known.extend(f"{name_dir.name}/{p.name}" for p in name_dir.iterdir() if p.is_dir())
    raise ConfigError(
        f"no run {ref!r} under {root}{did_you_mean(ref, known)}",
        hint="pass a run directory, a run name, or <name>/<run>; `dfwb runs list` shows them",
    )


def run_stamp(now: datetime.datetime | None = None) -> str:
    """``YYYYmmdd-HHMMSS`` in UTC, for a run directory's name."""
    moment = now or datetime.datetime.now(datetime.UTC)
    return moment.strftime("%Y%m%d-%H%M%S")


def data_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One row per source of a run's ``data.json``: role, name, protocol, split and counts."""
    rows: list[dict[str, Any]] = []
    for role in ("train", "val"):
        for source in data.get(role, []):
            counts: dict[str, Any] = {}
            index_sources = source.get("index", {}).get("sources", [])
            if index_sources:
                counts = index_sources[0].get("counts", {})
            protocol = source.get("protocol", {})
            rows.append(
                {
                    "role": role,
                    "name": source.get("name"),
                    "protocol": protocol.get("ref"),
                    "split": protocol.get("split"),
                    "in_split": counts.get("in_split"),
                    "included": counts.get("included"),
                }
            )
    return rows
