"""Run metadata: environment, git state and a command line without absolute paths."""

from __future__ import annotations

import datetime
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

from dfwb.core.paths import ROOT_ENV, ResolvedRoot, RootName, absolute
from dfwb.core.records._base import ABSOLUTE_PATH

__all__ = [
    "GitState",
    "RunInfo",
    "capture_env",
    "capture_git",
    "collect_run_info",
    "sanitize_command",
    "utc_now",
]


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True)
class RunInfo:
    """What every run and score file records about how it was produced."""

    env: dict[str, str | None]
    git: GitState | None
    command: str
    created: str
    plugins: dict[str, str | None] = field(default_factory=dict)  # provider -> version


def utc_now() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def capture_env() -> dict[str, str | None]:
    """Versions and device. Torch is inspected only if the caller already imported it."""
    from dfwb import __version__

    env: dict[str, str | None] = {
        "dfwb": __version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()}-{platform.machine()}",
        "torch": _version("torch"),
        "cuda": None,
        "device": None,
    }
    torch: Any = sys.modules.get("torch")
    if torch is not None:
        env["cuda"] = getattr(torch.version, "cuda", None)
        env["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return env


def capture_git(cwd: Path) -> GitState | None:
    """Commit and dirty flag of the repository containing ``cwd`` (``None`` outside git)."""

    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(cwd), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return GitState(commit, None if status is None else bool(status))


_PREFIXED = re.compile(r"^([A-Za-z][\w+.-]*:)((?:/(?!/)|~/).*)$")  # "run:/x", not "https://x"

# What ends a path found inside a larger token: whitespace, or a separator the absolute-path guard
# itself recognises in front of a path.
_PATH_END = re.compile(r"[\s=:,;\"'(]")
_FILE_URL = "file:///"


def _clean_path(text: str, roots: Mapping[RootName, ResolvedRoot]) -> str:
    """An absolute path (``/...`` or ``~/...``) as ``$DFWB_*_ROOT/...`` when it is under a root,
    else ``<abs>/<final component>``."""
    path = absolute(text)
    best: tuple[int, RootName, Path] | None = None
    for name, root in roots.items():
        deeper = best is None or (root.path is not None and len(root.path.parts) > best[0])
        if root.path is not None and path.is_relative_to(root.path) and deeper:
            best = (len(root.path.parts), name, root.path)
    if best is not None:
        relative = path.relative_to(best[2]).as_posix()
        return f"${ROOT_ENV[best[1]]}" + ("" if relative == "." else f"/{relative}")
    return f"<abs>/{path.name}"


def _clean(token: str, roots: Mapping[RootName, ResolvedRoot]) -> str:
    option, eq, value = token.partition("=")
    if eq and option and "/" not in option:  # --opt=value, or a config override key=value
        return f"{option}={_clean(value, roots)}"
    match = _PREFIXED.match(token)
    if match:
        return match.group(1) + _clean(match.group(2), roots)
    if not (token.startswith("/") or token.startswith("~/")):
        return token
    return _clean_path(token, roots)


def _scrub(text: str, roots: Mapping[RootName, ResolvedRoot]) -> str:
    """``text`` with every absolute path the record guard would still find in it cleaned as well:
    a path after a space, a comma or a quote inside one argument, in a ``file:`` URL, after a
    ``key=`` whose key holds a slash, and so on. Each is cut at the next separator and cleaned
    like a whole-argument path. A replacement starts with ``$`` or ``<`` and holds no separator,
    so it is never flagged itself: every pass removes at least one flagged place, and the loop
    ends when the guard finds none."""
    for _ in range(len(text) + 1):
        match = ABSOLUTE_PATH.search(text)
        if match is None:
            break
        found = match.group(0)
        if found == _FILE_URL:
            start = match.start() + len(_FILE_URL) - 1
        else:
            start = match.end() - (2 if found.endswith("~/") else 1)
        end_match = _PATH_END.search(text, start + 1)
        end = len(text) if end_match is None else end_match.start()
        text = text[:start] + _clean_path(text[start:end], roots) + text[end:]
    return text


def sanitize_command(argv: Sequence[str], roots: Mapping[RootName, ResolvedRoot]) -> str:
    """The command line without absolute paths: paths under a root become ``$DFWB_*_ROOT/...``,
    other absolute paths keep only their final component. This holds for a path anywhere inside
    an argument, not only one that starts it, using the very pattern
    :func:`~dfwb.core.records.assert_no_absolute_paths` checks records with, so a record that
    stores the result (a run's ``env.json``, a score file's meta) always passes that check."""
    if not argv:
        return ""
    program = Path(argv[0]).name
    cleaned = shlex.join([program, *(_scrub(_clean(arg, roots), roots) for arg in argv[1:])])
    return _scrub(cleaned, roots)


def collect_run_info(
    argv: Sequence[str] | None = None,
    *,
    cwd: Path | None = None,
    roots: Mapping[RootName, ResolvedRoot] | None = None,
) -> RunInfo:
    """Gather :class:`RunInfo` for the current process."""
    from dfwb.core.paths import resolve_roots
    from dfwb.core.plugins import PluginStatus, load_plugins

    cwd = Path.cwd() if cwd is None else cwd
    roots = resolve_roots(cwd=cwd) if roots is None else roots
    report = load_plugins()
    plugins = {r.provider: r.version for r in report.records if r.status is PluginStatus.OK}
    return RunInfo(
        env=capture_env(),
        git=capture_git(cwd),
        command=sanitize_command(sys.argv if argv is None else argv, roots),
        created=utc_now(),
        plugins=plugins,
    )
