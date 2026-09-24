"""Filesystem roots and root-relative paths.

Four roots hold everything DFWB reads or writes. Each is resolved in this order:
CLI flag > environment variable > ``./dfwb.toml`` ``[roots]`` > user ``config.toml`` ``[roots]`` >
default. Published and shared artifacts never store absolute paths; they store a
:class:`RelPath` (root name + POSIX path relative to that root).
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, get_args

import platformdirs

from dfwb.core.errors import ConfigError, ContractError, did_you_mean

__all__ = [
    "PROJECT_CONFIG",
    "ROOT_ENV",
    "RelPath",
    "ResolvedRoot",
    "RootName",
    "absolute",
    "relativize",
    "require_root",
    "resolve_roots",
    "user_config_path",
]

RootName = Literal["datasets", "work", "cache", "runs"]
Source = Literal["flag", "env", "project", "user", "default", "unset"]

ROOT_NAMES: Final[tuple[RootName, ...]] = get_args(RootName)
ROOT_ENV: Final[dict[RootName, str]] = {
    "datasets": "DFWB_DATASETS_ROOT",
    "work": "DFWB_WORK_ROOT",
    "cache": "DFWB_CACHE_ROOT",
    "runs": "DFWB_RUNS_ROOT",
}
PROJECT_CONFIG: Final = "dfwb.toml"


@dataclass(frozen=True)
class ResolvedRoot:
    """A root's value and where it came from (``dfwb doctor`` prints both)."""

    name: RootName
    path: Path | None
    source: Source
    detail: str
    warning: str | None = None


def user_config_path() -> Path:
    """``~/.config/dfwb/config.toml`` (respects ``XDG_CONFIG_HOME``)."""
    return Path(platformdirs.user_config_dir("dfwb")) / "config.toml"


def absolute(value: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Absolute, normalised path (``~`` expanded, ``..`` folded); symlinks are kept as written.

    Relative paths are taken relative to ``base`` (default: the current directory).
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() if base is None else base) / path
    return Path(os.path.normpath(path))


def _read_roots(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path}: invalid TOML: {exc}", hint="fix the file or remove it"
        ) from None
    roots = data.get("roots", {})
    if not isinstance(roots, dict):
        raise ConfigError(
            f"{path}: [roots] must be a table", hint='e.g. [roots]\ndatasets = "/data"'
        )
    for key, value in roots.items():
        if key not in ROOT_NAMES:
            raise ConfigError(
                f"{path}: unknown root {key!r}{did_you_mean(key, ROOT_NAMES)}",
                hint="roots: " + ", ".join(ROOT_NAMES),
            )
        if not isinstance(value, str) or not value:
            raise ConfigError(
                f"{path}: roots.{key} must be a non-empty string", hint="quote the path"
            )
    return dict(roots)


def resolve_roots(
    *,
    flags: Mapping[str, str | os.PathLike[str] | None] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    user_config: Path | None = None,
) -> dict[RootName, ResolvedRoot]:
    """Resolve all four roots. Nothing is created on disk.

    Args:
        flags: Values given on the command line, keyed by root name.
        env: Environment to read (defaults to ``os.environ``).
        cwd: Directory holding the project ``dfwb.toml`` (defaults to the current directory).
        user_config: User config file (defaults to :func:`user_config_path`).
    """
    flags = flags or {}
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    project_file = cwd / PROJECT_CONFIG
    user_file = user_config_path() if user_config is None else user_config
    project = _read_roots(project_file)
    user = _read_roots(user_file)

    resolved: dict[RootName, ResolvedRoot] = {}
    for name in ROOT_NAMES:
        variable = ROOT_ENV[name]
        flag = flags.get(name)
        if flag:
            resolved[name] = ResolvedRoot(name, absolute(flag, cwd), "flag", f"--{name}-root")
        elif env.get(variable):
            resolved[name] = ResolvedRoot(name, absolute(env[variable], cwd), "env", variable)
        elif name in project:
            path = absolute(project[name], project_file.parent)
            resolved[name] = ResolvedRoot(name, path, "project", str(project_file))
        elif name in user:
            path = absolute(user[name], user_file.parent)
            resolved[name] = ResolvedRoot(name, path, "user", str(user_file))
        else:
            resolved[name] = _default(name, resolved, cwd)
    return resolved


def _default(name: RootName, resolved: Mapping[RootName, ResolvedRoot], cwd: Path) -> ResolvedRoot:
    if name == "datasets":
        return ResolvedRoot(name, None, "unset", ROOT_ENV[name])
    if name == "work":
        datasets = resolved["datasets"].path
        if datasets is None:
            return ResolvedRoot(name, None, "unset", ROOT_ENV[name])
        path = datasets.parent / "dfwb-work"
        warning = f"{ROOT_ENV['work']} is not set; using {path} (next to {ROOT_ENV['datasets']})"
        return ResolvedRoot(name, path, "default", "next to the datasets root", warning)
    if name == "cache":
        return ResolvedRoot(
            name, Path(platformdirs.user_cache_dir("dfwb")), "default", "user cache"
        )
    return ResolvedRoot(name, absolute(cwd / "runs"), "default", "./runs")


def require_root(name: RootName, roots: Mapping[RootName, ResolvedRoot]) -> Path:
    """The path of root ``name``, or a :class:`ConfigError` explaining how to set it."""
    path = roots[name].path
    if path is None:
        variable = ROOT_ENV[name]
        raise ConfigError(
            f"{variable} is not set",
            hint=f"export {variable}=/path/to/{name} or add it under [roots] in ./{PROJECT_CONFIG}",
        )
    return path


@dataclass(frozen=True)
class RelPath:
    """A path relative to a named root, e.g. ``RelPath("runs", "vit/2026-01-01-s42")``."""

    root: RootName
    path: str

    def __post_init__(self) -> None:
        if self.root not in ROOT_NAMES:
            raise ContractError(
                f"unknown root {self.root!r}", hint="roots: " + ", ".join(ROOT_NAMES)
            )
        pure = PurePosixPath(self.path)
        if not self.path or pure.is_absolute() or ".." in pure.parts or "\\" in self.path:
            raise ContractError(
                f"{self.path!r} is not a relative POSIX path",
                hint="store paths relative to a DFWB root, without '..'",
            )
        object.__setattr__(self, "path", pure.as_posix())

    def __str__(self) -> str:
        return f"{self.root}:{self.path}"

    def resolve(self, roots: Mapping[RootName, ResolvedRoot]) -> Path:
        """The absolute local path."""
        return require_root(self.root, roots) / self.path


def relativize(path: Path, roots: Mapping[RootName, ResolvedRoot]) -> RelPath:
    """Express an absolute ``path`` relative to the deepest root that contains it."""
    target = absolute(path)
    best: tuple[int, RootName, Path] | None = None
    for name, root in roots.items():
        if root.path is not None and target.is_relative_to(root.path):
            depth = len(root.path.parts)
            if best is None or depth > best[0]:
                best = (depth, name, root.path)
    if best is None or target == best[2]:
        raise ContractError(
            f"{target} is not inside any DFWB root",
            hint="place it under one of the roots shown by `dfwb doctor`",
        )
    return RelPath(best[1], target.relative_to(best[2]).as_posix())
