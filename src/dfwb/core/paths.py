"""Filesystem roots and root-relative paths.

Four roots hold everything DFWB reads or writes. Each is resolved in this order:
CLI flag > environment variable > ``./dfwb.toml`` ``[roots]`` > user ``config.toml`` ``[roots]`` >
default. Published and shared artifacts never store absolute paths; they store a
:class:`RelPath` (root name + POSIX path relative to that root).
"""

from __future__ import annotations

import os
import socket
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
    "DatasetLocation",
    "RelPath",
    "ResolvedRoot",
    "RootName",
    "absolute",
    "current_host",
    "dataset_overrides",
    "locate_dataset",
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
HOST_ENV: Final = "DFWB_HOST"
DATASET_ENV_PREFIX: Final = "DFWB_DATASET_"
_HOST_TABLE_KEYS: Final = ("roots", "datasets")


@dataclass(frozen=True)
class ResolvedRoot:
    """A root's value and where it came from (``dfwb doctor`` prints both)."""

    name: RootName
    path: Path | None
    source: Source
    detail: str
    warning: str | None = None
    paths: tuple[Path, ...] = ()


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


def current_host(env: Mapping[str, str] | None = None) -> str:
    """The current host's short, lower-case name.

    ``env[DFWB_HOST]`` wins when set; otherwise :func:`socket.gethostname`, cut at the first
    ``.`` to drop any domain suffix.
    """
    env = os.environ if env is None else env
    value = env.get(HOST_ENV)
    if value:
        return value.lower()
    return socket.gethostname().split(".", 1)[0].lower()


def _split_list(value: str) -> list[str]:
    """Split a ``os.pathsep``-separated value, dropping empty entries."""
    return [part for part in value.split(os.pathsep) if part]


def _dataset_env_var(dataset_id: str) -> str:
    return DATASET_ENV_PREFIX + dataset_id.upper().replace("-", "_")


def _read_roots(table: object, path: Path) -> dict[str, str | list[str]]:
    """Validate a ``[roots]``-shaped table. ``datasets`` may be a string or list of strings."""
    if not isinstance(table, dict):
        raise ConfigError(
            f"{path}: [roots] must be a table", hint='e.g. [roots]\ndatasets = "/data"'
        )
    result: dict[str, str | list[str]] = {}
    for key, value in table.items():
        if key not in ROOT_NAMES:
            raise ConfigError(
                f"{path}: unknown root {key!r}{did_you_mean(key, ROOT_NAMES)}",
                hint="roots: " + ", ".join(ROOT_NAMES),
            )
        if key == "datasets":
            result[key] = _read_datasets_root_value(value, path)
        elif not isinstance(value, str) or not value:
            raise ConfigError(
                f"{path}: roots.{key} must be a non-empty string", hint="quote the path"
            )
        else:
            result[key] = value
    return result


def _read_datasets_root_value(value: object, path: Path) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value and all(isinstance(v, str) and v for v in value):
        return list(value)
    raise ConfigError(
        f"{path}: roots.datasets must be a non-empty string or a list of strings",
        hint="quote the path(s)",
    )


def _read_datasets_table(table: object, path: Path, label: str) -> dict[str, str]:
    """Validate a ``[datasets]``-shaped table: dataset id -> folder path."""
    if not isinstance(table, dict):
        raise ConfigError(
            f"{path}: [{label}] must be a table", hint='e.g. [datasets]\nkodf = "/data/KoDF"'
        )
    result: dict[str, str] = {}
    for key, value in table.items():
        if not isinstance(value, str) or not value:
            raise ConfigError(
                f"{path}: {label}.{key} must be a non-empty string", hint="quote the path"
            )
        result[key] = value
    return result


@dataclass(frozen=True)
class _FileSettings:
    """One config file's ``[roots]``/``[datasets]`` and the current host's overrides of them."""

    roots: dict[str, str | list[str]]
    datasets: dict[str, str]
    host_roots: dict[str, str | list[str]]
    host_datasets: dict[str, str]
    file: Path


def _read_file(path: Path, host: str) -> _FileSettings:
    """Read ``[roots]``, ``[datasets]`` and ``[hosts.<host>]`` from one TOML file, once."""
    if not path.is_file():
        return _FileSettings({}, {}, {}, {}, path)
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path}: invalid TOML: {exc}", hint="fix the file or remove it"
        ) from None

    roots = _read_roots(data.get("roots", {}), path)
    datasets = _read_datasets_table(data.get("datasets", {}), path, "datasets")

    hosts = data.get("hosts", {})
    if not isinstance(hosts, dict):
        raise ConfigError(f"{path}: [hosts] must be a table", hint="e.g. [hosts.myhost.roots]")

    host_roots: dict[str, str | list[str]] = {}
    host_datasets: dict[str, str] = {}
    host_table = hosts.get(host)
    if isinstance(host_table, dict):
        for key in host_table:
            if key not in _HOST_TABLE_KEYS:
                raise ConfigError(
                    f"{path}: unknown key {key!r} in [hosts.{host}]"
                    f"{did_you_mean(key, _HOST_TABLE_KEYS)}",
                    hint="use [hosts.<host>.roots] and [hosts.<host>.datasets]",
                )
        host_roots = _read_roots(host_table.get("roots", {}), path)
        host_datasets = _read_datasets_table(
            host_table.get("datasets", {}), path, f"hosts.{host}.datasets"
        )
    return _FileSettings(roots, datasets, host_roots, host_datasets, path)


def resolve_roots(
    *,
    flags: Mapping[str, str | os.PathLike[str] | None] | None = None,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    user_config: Path | None = None,
) -> dict[RootName, ResolvedRoot]:
    """Resolve all four roots. Nothing is created on disk.

    Precedence: flag > environment > project host table > project > user host table > user >
    default.

    Args:
        flags: Values given on the command line, keyed by root name.
        env: Environment to read (defaults to ``os.environ``).
        cwd: Directory holding the project ``dfwb.toml`` (defaults to the current directory).
        user_config: User config file (defaults to :func:`user_config_path`).
    """
    flags = flags or {}
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    host = current_host(env)
    project_file = cwd / PROJECT_CONFIG
    user_file = user_config_path() if user_config is None else user_config
    project = _read_file(project_file, host)
    user = _read_file(user_file, host)

    resolved: dict[RootName, ResolvedRoot] = {}
    for name in ROOT_NAMES:
        resolved[name] = _resolve_one(name, flags, env, cwd, host, project, user, resolved)
    return resolved


def _values_of(name: RootName, raw: str | list[str]) -> list[str]:
    return raw if isinstance(raw, list) else [raw]


def _resolve_one(
    name: RootName,
    flags: Mapping[str, str | os.PathLike[str] | None],
    env: Mapping[str, str],
    cwd: Path,
    host: str,
    project: _FileSettings,
    user: _FileSettings,
    resolved: Mapping[RootName, ResolvedRoot],
) -> ResolvedRoot:
    variable = ROOT_ENV[name]
    flag = flags.get(name)
    if flag:
        values = _split_list(str(flag)) if name == "datasets" else [str(flag)]
        paths = tuple(absolute(v, cwd) for v in values)
        return ResolvedRoot(name, paths[0], "flag", f"--{name}-root", paths=paths)
    env_value = env.get(variable)
    if env_value:
        values = _split_list(env_value) if name == "datasets" else [env_value]
        paths = tuple(absolute(v, cwd) for v in values)
        return ResolvedRoot(name, paths[0], "env", variable, paths=paths)
    if name in project.host_roots:
        values = _values_of(name, project.host_roots[name])
        paths = tuple(absolute(v, project.file.parent) for v in values)
        detail = f"{project.file} [hosts.{host}.roots]"
        return ResolvedRoot(name, paths[0], "project", detail, paths=paths)
    if name in project.roots:
        values = _values_of(name, project.roots[name])
        paths = tuple(absolute(v, project.file.parent) for v in values)
        return ResolvedRoot(name, paths[0], "project", str(project.file), paths=paths)
    if name in user.host_roots:
        values = _values_of(name, user.host_roots[name])
        paths = tuple(absolute(v, user.file.parent) for v in values)
        detail = f"{user.file} [hosts.{host}.roots]"
        return ResolvedRoot(name, paths[0], "user", detail, paths=paths)
    if name in user.roots:
        values = _values_of(name, user.roots[name])
        paths = tuple(absolute(v, user.file.parent) for v in values)
        return ResolvedRoot(name, paths[0], "user", str(user.file), paths=paths)
    return _default(name, resolved, cwd)


def _default(name: RootName, resolved: Mapping[RootName, ResolvedRoot], cwd: Path) -> ResolvedRoot:
    if name == "datasets":
        return ResolvedRoot(name, None, "unset", ROOT_ENV[name])
    if name == "work":
        datasets_root = resolved["datasets"]
        if datasets_root.path is None:
            return ResolvedRoot(name, None, "unset", ROOT_ENV[name])
        path = datasets_root.path.parent / "dfwb-work"
        warning = f"{ROOT_ENV['work']} is not set; using {path} (next to {ROOT_ENV['datasets']})"
        if len(datasets_root.paths) > 1:
            warning += "; set DFWB_WORK_ROOT explicitly when datasets span several locations"
        return ResolvedRoot(name, path, "default", "next to the datasets root", warning, (path,))
    if name == "cache":
        path = Path(platformdirs.user_cache_dir("dfwb"))
        return ResolvedRoot(name, path, "default", "user cache", paths=(path,))
    path = absolute(cwd / "runs")
    return ResolvedRoot(name, path, "default", "./runs", paths=(path,))


def dataset_overrides(
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    user_config: Path | None = None,
) -> dict[str, tuple[Path, str]]:
    """Every dataset folder override: dataset id -> (folder, where it came from).

    Sources, highest precedence first: ``DFWB_DATASET_<ID>`` environment variables, the project
    host table, the project ``[datasets]`` table, the user host table, the user ``[datasets]``
    table.
    """
    env = os.environ if env is None else env
    cwd = Path.cwd() if cwd is None else cwd
    host = current_host(env)
    project_file = cwd / PROJECT_CONFIG
    user_file = user_config_path() if user_config is None else user_config
    project = _read_file(project_file, host)
    user = _read_file(user_file, host)

    found: dict[str, tuple[Path, str]] = {}
    for dataset_id, value in user.datasets.items():
        found[dataset_id] = (absolute(value, user.file.parent), str(user.file))
    for dataset_id, value in user.host_datasets.items():
        found[dataset_id] = (
            absolute(value, user.file.parent),
            f"{user.file} [hosts.{host}.datasets]",
        )
    for dataset_id, value in project.datasets.items():
        found[dataset_id] = (absolute(value, project.file.parent), str(project.file))
    for dataset_id, value in project.host_datasets.items():
        found[dataset_id] = (
            absolute(value, project.file.parent),
            f"{project.file} [hosts.{host}.datasets]",
        )

    for dataset_id in list(found):
        variable = _dataset_env_var(dataset_id)
        if env.get(variable):
            found[dataset_id] = (absolute(env[variable], cwd), f"env: {variable}")
    for key, value in env.items():
        if key.startswith(DATASET_ENV_PREFIX) and value:
            dataset_id = key[len(DATASET_ENV_PREFIX) :].lower().replace("_", "-")
            if dataset_id not in found:
                found[dataset_id] = (absolute(value, cwd), f"env: {key}")
    return found


@dataclass(frozen=True)
class DatasetLocation:
    """Where a dataset's folder was found (or came from an override)."""

    dataset_id: str
    path: Path
    root: Path | None
    source: str
    also_found: tuple[Path, ...] = ()


def locate_dataset(
    dataset_id: str,
    folder: str,
    roots: Mapping[RootName, ResolvedRoot],
    *,
    overrides: Mapping[str, tuple[Path, str]] | None = None,
) -> DatasetLocation:
    """Find dataset ``dataset_id``'s folder, named ``folder``, under one of the datasets roots.

    An entry in ``overrides`` wins outright. Otherwise the first datasets root that holds
    ``folder`` as a directory is used, and any later roots that hold it too are reported in
    ``also_found``.
    """
    if overrides is not None and dataset_id in overrides:
        path, source = overrides[dataset_id]
        if not path.is_dir():
            raise ConfigError(
                f"{dataset_id}: {path} is not a directory",
                hint=f"check {source}, or point {_dataset_env_var(dataset_id)} at the dataset",
            )
        return DatasetLocation(dataset_id, path, None, source)

    datasets_root = roots.get("datasets")
    search_paths = datasets_root.paths if datasets_root is not None else ()
    matches = [(root, root / folder) for root in search_paths if (root / folder).is_dir()]
    if not matches:
        searched = ", ".join(str(p) for p in search_paths) or "no datasets root is set"
        raise ConfigError(
            f"{dataset_id} ({folder!r}) was not found in any datasets root: searched {searched}",
            hint=f"set {_dataset_env_var(dataset_id)}=/path/to/{folder}, "
            "or add it under [datasets]",
        )
    first_root, first_path = matches[0]
    also_found = tuple(path for _, path in matches[1:])
    return DatasetLocation(dataset_id, first_path, first_root, "search", also_found=also_found)


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
