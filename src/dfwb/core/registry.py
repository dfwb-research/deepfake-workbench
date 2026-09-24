"""Registries (contract C1): lazy, provider-aware, collision-safe lookup of pluggable components.

A registry maps lower-kebab-case keys to *import-path targets* (``"package.module:Attr"``) plus
metadata. Nothing is imported until :meth:`Registry.load` or :meth:`Registry.build` is called, so
registering is cheap and listing never needs torch. Two providers may register the same key; the
plain key then becomes ambiguous and must be qualified as ``<provider>:<key>``.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.resources
import importlib.util
import inspect
import re
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model

from dfwb.core.errors import (
    AmbiguousKeyError,
    ConfigError,
    InstallationError,
    Loc,
    PluginError,
    UnknownKeyError,
    did_you_mean,
    format_loc,
    model_fields_at,
    validation_problems,
)

__all__ = ["BUILTIN_PROVIDER", "LOCAL_PROVIDER", "Entry", "Registry", "canonical_name", "providing"]

RegistryKind = Literal["code", "data"]

BUILTIN_PROVIDER = "dfwb"
LOCAL_PROVIDER = "local"

_KEY = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_CODE_TARGET = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_DATA_TARGET = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[\w.+-]+(?:/[\w.+-]+)*$")
_IMPORT_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def canonical_name(name: str) -> str:
    """Normalise a distribution name as PEP 503 does: ``Dfwb_Torch.SRM`` -> ``dfwb-torch-srm``."""
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Entry:
    """One registration: a key, where its object lives, who provided it and what it needs."""

    registry: str
    key: str
    target: str
    provider: str
    summary: str
    aliases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    params: str | None = None
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def qualified_key(self) -> str:
        """``<provider>:<key>``, which is never ambiguous."""
        return f"{self.provider}:{self.key}"


_provider: ContextVar[str] = ContextVar("dfwb_registry_provider", default=LOCAL_PROVIDER)
_recorder: ContextVar[list[tuple[Registry[Any], Entry]] | None] = ContextVar(
    "dfwb_registry_recorder", default=None
)


@contextmanager
def providing(provider: str) -> Iterator[list[tuple[Registry[Any], Entry]]]:
    """Attribute registrations made inside the block to ``provider``.

    Yields the list of ``(registry, entry)`` pairs added inside the block, so a caller can roll
    them back if the block fails.
    """
    added: list[tuple[Registry[Any], Entry]] = []
    provider_token = _provider.set(provider)
    recorder_token = _recorder.set(added)
    try:
        yield added
    finally:
        _recorder.reset(recorder_token)
        _provider.reset(provider_token)


@functools.cache
def _catalogue() -> dict[str, Any]:
    text = importlib.resources.files("dfwb.core").joinpath("catalogue.toml").read_text("utf-8")
    return tomllib.loads(text)


def catalogue_requirement(registry: str, key: str) -> str | None:
    """The pip requirement that provides ``registry/key`` according to the shipped catalogue."""
    value = _catalogue().get("keys", {}).get(registry, {}).get(key)
    return str(value) if value is not None else None


def install_hint(import_name: str) -> str:
    """How to install the package that provides ``import_name``."""
    top = import_name.partition(".")[0]
    extra = _catalogue().get("imports", {}).get(top)
    if extra is not None:
        return f'pip install "deepfake-workbench[{extra}]"'
    return f"install the package that provides the {top!r} module"


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__


def _is_installed(import_name: str) -> bool:
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


@dataclass(frozen=True)
class _ParamSpec:
    """How to validate the keyword arguments of one entry."""

    allowed: frozenset[str] | None  # None: any keyword is accepted (target takes **kwargs)
    model: type[BaseModel] | None  # validates by alias = parameter name
    adapter: TypeAdapter[Any] | None  # for dataclass / TypedDict params models

    def fields_at(self, loc: Loc) -> Iterable[str]:
        """Valid keys at ``loc`` inside the params (for did-you-mean on nested sections)."""
        if not loc and self.allowed is not None:
            return self.allowed
        return model_fields_at(self.model, loc) if self.model is not None else ()


class Registry[T]:
    """A named registry of lazily imported components.

    Args:
        name: Registry name, e.g. ``"layers"``.
        kind: ``"code"`` targets are ``module:attr`` import paths; ``"data"`` targets are
            ``package:relative/path`` resources inside an installed package (no code import).
        on_access: Called before every read (used to load plugins on first access).
        unknown_hint: Called with ``(registry, key)`` to explain an unknown key.
    """

    def __init__(
        self,
        name: str,
        *,
        kind: RegistryKind = "code",
        on_access: Callable[[], object] | None = None,
        unknown_hint: Callable[[str, str], str | None] | None = None,
    ) -> None:
        self.name = name
        self.kind: RegistryKind = kind
        self._on_access = on_access
        self._unknown_hint = unknown_hint
        self._entries: list[Entry] = []
        self._loaded: dict[tuple[str, str], Any] = {}
        self._param_specs: dict[tuple[str, str], _ParamSpec] = {}

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, entries={len(self._entries)})"

    # ------------------------------------------------------------------ registration

    def add(
        self,
        key: str,
        target: str,
        *,
        summary: str,
        aliases: Sequence[str] = (),
        requires: Sequence[str] = (),
        params: str | None = None,
        **meta: Any,
    ) -> None:
        """Register ``key`` -> ``target`` for the current provider. Never imports anything."""
        entry = self._make_entry(key, target, summary, aliases, requires, params, meta)
        self._insert(entry)

    def register(
        self,
        key: str,
        *,
        summary: str,
        aliases: Sequence[str] = (),
        requires: Sequence[str] = (),
        params: str | None = None,
        **meta: Any,
    ) -> Callable[[T], T]:
        """Decorator form of :meth:`add` for objects that are already imported."""

        def decorator(obj: T) -> T:
            qualname = getattr(obj, "__qualname__", None)
            module = getattr(obj, "__module__", None)
            if not isinstance(qualname, str) or not isinstance(module, str) or "<" in qualname:
                raise PluginError(
                    f"{self.name}: register({key!r}) can only decorate module-level classes "
                    "and functions",
                    hint=f"use {self.name}.add({key!r}, target='package.module:Name', ...)",
                )
            entry = self._make_entry(
                key, f"{module}:{qualname}", summary, aliases, requires, params, meta
            )
            self._insert(entry)
            self._loaded[(entry.provider, entry.key)] = obj
            return obj

        return decorator

    def _make_entry(
        self,
        key: str,
        target: str,
        summary: str,
        aliases: Sequence[str],
        requires: Sequence[str],
        params: str | None,
        meta: Mapping[str, Any],
    ) -> Entry:
        where = f"{self.name}/{key}"
        if not isinstance(key, str) or not _KEY.match(key):
            raise PluginError(
                f"{self.name}: invalid key {key!r}",
                hint="keys are lower-kebab-case, e.g. 'my-layer'",
            )
        pattern = _CODE_TARGET if self.kind == "code" else _DATA_TARGET
        if not isinstance(target, str) or not pattern.match(target) or ".." in target:
            example = "'package.module:Name'" if self.kind == "code" else "'package:dir/file.yaml'"
            raise PluginError(f"{where}: invalid target {target!r}", hint=f"use e.g. {example}")
        if not isinstance(summary, str) or not summary.strip() or "\n" in summary:
            raise PluginError(
                f"{where}: summary must be one non-empty line", hint="pass summary='...'"
            )
        clean_aliases: list[str] = []
        for alias in aliases:
            norm = str(alias).strip().lower()
            if not norm or ":" in norm or any(c.isspace() for c in norm):
                raise PluginError(
                    f"{where}: invalid alias {alias!r}",
                    hint="aliases are non-empty and contain no spaces or ':'",
                )
            if norm != key and norm not in clean_aliases:
                clean_aliases.append(norm)
        for name in requires:
            if not isinstance(name, str) or not _IMPORT_NAME.match(name):
                raise PluginError(
                    f"{where}: invalid requires entry {name!r}", hint="use import names"
                )
        if params is not None and not _CODE_TARGET.match(params):
            raise PluginError(f"{where}: invalid params {params!r}", hint="use 'module:Model'")
        return Entry(
            registry=self.name,
            key=key,
            target=target,
            provider=_provider.get(),
            summary=summary.strip(),
            aliases=tuple(clean_aliases),
            requires=tuple(requires),
            params=params,
            meta=MappingProxyType(dict(meta)),
        )

    def _insert(self, entry: Entry) -> None:
        names = {entry.key, *entry.aliases}
        for other in self._entries:
            if other.provider == entry.provider and names & {other.key, *other.aliases}:
                raise PluginError(
                    f"{self.name}/{entry.key}: {entry.provider} registers "
                    f"{sorted(names & {other.key, *other.aliases})} twice",
                    hint="each key and alias may be registered once per provider",
                )
        self._entries.append(entry)
        recorder = _recorder.get()
        if recorder is not None:
            recorder.append((self, entry))

    def _remove(self, entry: Entry) -> None:
        self._entries = [e for e in self._entries if e is not entry]
        self._loaded.pop((entry.provider, entry.key), None)
        self._param_specs.pop((entry.provider, entry.key), None)

    def _clear(self) -> None:
        self._entries.clear()
        self._loaded.clear()
        self._param_specs.clear()

    # ------------------------------------------------------------------ lookup

    def _ensure_loaded(self) -> None:
        if self._on_access is not None:
            self._on_access()

    def _candidates(self, key: str) -> list[Entry]:
        provider, sep, name = key.partition(":")
        if not sep:
            provider, name = "", provider
        wanted = name.strip().lower()
        wanted_provider = canonical_name(provider) if sep else None
        return [
            e
            for e in self._entries
            if (wanted_provider is None or e.provider == wanted_provider)
            and (e.key == wanted or wanted in e.aliases)
        ]

    def entry(self, key: str) -> Entry:
        """Resolve ``key`` (case-insensitive, alias or ``provider:key``) to its entry."""
        self._ensure_loaded()
        found = self._candidates(key)
        if len(found) == 1:
            return found[0]
        if found:
            qualified = sorted(e.qualified_key for e in found)
            raise AmbiguousKeyError(
                f"{self.name} {key!r} is ambiguous: provided by {', '.join(qualified)}",
                hint=f"use a qualified key, e.g. {qualified[0]!r}",
            )
        names = [n for e in self._entries for n in (e.key, *e.aliases)]
        wanted = key.partition(":")[2] if ":" in key else key
        message = f"{self.name}: unknown key {key!r}{did_you_mean(wanted.lower(), names)}"
        hint = self._unknown_hint(self.name, wanted.lower()) if self._unknown_hint else None
        raise UnknownKeyError(
            message, hint=hint or f"run `dfwb plugins list` to see the registered {self.name}"
        )

    def keys(self) -> list[str]:
        """Sorted plain keys (without aliases); a key provided twice appears once."""
        self._ensure_loaded()
        return sorted({e.key for e in self._entries})

    def entries(self) -> list[Entry]:
        """All entries, sorted by key then provider."""
        self._ensure_loaded()
        return sorted(self._entries, key=lambda e: (e.key, e.provider))

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        self._ensure_loaded()
        return bool(self._candidates(key))

    # ------------------------------------------------------------------ loading

    def load(self, key: str) -> T:
        """Import (code) or locate (data) the target of ``key``; the result is cached."""
        entry = self.entry(key)
        cache_key = (entry.provider, entry.key)
        if cache_key not in self._loaded:
            self._check_requires(entry)
            if self.kind == "code":
                self._loaded[cache_key] = _import_target(entry, entry.target)
            else:
                self._loaded[cache_key] = _resource_target(entry)
        obj: T = self._loaded[cache_key]
        return obj

    def validate(self, key: str, /, **params: Any) -> dict[str, Any]:
        """Validate keyword ``params`` for ``key`` without calling the target.

        Returns the validated keyword arguments (only the keys that were passed, coerced to the
        declared types). Raises :class:`ConfigError` for unknown, missing or ill-typed params.
        """
        entry = self.entry(key)
        if self.kind != "code":
            raise PluginError(
                f"{self.name}/{entry.key} is a data entry and takes no parameters",
                hint=f"use {self.name}.load({key!r})",
            )
        spec = self._param_spec(entry)
        where = f"{self.name}/{entry.key}"
        if spec.allowed is not None:
            for name in params:
                if name not in spec.allowed:
                    problem = f"unknown parameter{did_you_mean(name, spec.allowed)}"
                    raise ConfigError(
                        f"{where}: {name}: {problem}",
                        hint="accepted parameters: " + (", ".join(sorted(spec.allowed)) or "none"),
                        problems=[((name,), problem)],
                    )
        try:
            if spec.model is not None:
                validated = spec.model.model_validate(params)
                values = {
                    field_info.alias or name: getattr(validated, name)
                    for name, field_info in type(validated).model_fields.items()
                }
                values.update(validated.model_extra or {})
            elif spec.adapter is not None:
                result = spec.adapter.validate_python(params)
                values = result if isinstance(result, dict) else _vars_of(result)
            else:
                values = dict(params)
        except ValidationError as exc:
            problems = validation_problems(exc, fields_at=spec.fields_at)
            raise ConfigError(
                f"{where}: " + "; ".join(f"{format_loc(loc)}: {text}" for loc, text in problems),
                hint=f"see `dfwb plugins info {self.name}/{entry.key}`",
                problems=problems,
            ) from None
        return {name: values[name] for name in params}

    def build(self, key: str, /, **params: Any) -> Any:
        """Resolve ``key``, import the target, validate ``params`` and call the target."""
        kwargs = self.validate(key, **params)
        target = self.load(key)
        if not callable(target):
            raise PluginError(
                f"{self.name}/{key}: target is not callable", hint="register a class or a factory"
            )
        return target(**kwargs)

    def _check_requires(self, entry: Entry) -> None:
        for name in entry.requires:
            if not _is_installed(name):
                raise InstallationError(
                    f"{entry.registry}/{entry.key} needs {name!r}, which is not installed",
                    hint=install_hint(name),
                )

    def _param_spec(self, entry: Entry) -> _ParamSpec:
        cache_key = (entry.provider, entry.key)
        if cache_key not in self._param_specs:
            if entry.params is not None:  # a params model validates without the (heavy) target
                spec = _spec_from_params_model(entry, _import_target(entry, entry.params))
            else:
                spec = _spec_from_signature(entry, self.load(entry.qualified_key))
            self._param_specs[cache_key] = spec
        return self._param_specs[cache_key]


def _vars_of(obj: Any) -> dict[str, Any]:
    """Field values of a dataclass instance (shallow)."""
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _import_target(entry: Entry, target: str) -> Any:
    module_name, _, attr_path = target.partition(":")
    where = f"{entry.registry}/{entry.key}"
    try:
        obj: Any = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        if module_name == missing or module_name.startswith(missing + "."):
            raise PluginError(
                f"{where}: module {module_name!r} not found (provided by {entry.provider})",
                hint=f"reinstall {entry.provider}, or check the target in its register()",
            ) from None
        raise InstallationError(
            f"{where}: importing {module_name!r} failed: no module named {missing!r}",
            hint=install_hint(missing),
        ) from None
    except Exception as exc:
        raise PluginError(
            f"{where}: importing {module_name!r} raised {_first_line(exc)}",
            hint=f"this is a bug in {entry.provider}; run with --debug for the traceback",
        ) from exc
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise PluginError(
                f"{where}: {target!r} does not exist (provided by {entry.provider})",
                hint=f"check the target registered by {entry.provider}",
            ) from None
    return obj


def _resource_target(entry: Entry) -> Path:
    """Locate ``package:relative/path`` without importing the package (data only, C1)."""
    package, _, relative = entry.target.partition(":")
    top, *inner = package.split(".")
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        spec = None
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locations:
        raise PluginError(
            f"{entry.registry}/{entry.key}: package {package!r} not found",
            hint=f"reinstall {entry.provider}",
        )
    resource = Path(locations[0]).joinpath(*inner, *relative.split("/"))
    if not (resource.is_dir() or resource.is_file()):
        raise PluginError(
            f"{entry.registry}/{entry.key}: resource {entry.target!r} does not exist",
            hint=f"reinstall {entry.provider}, or check the target in its register()",
        )
    return resource


def _spec_from_params_model(entry: Entry, model: Any) -> _ParamSpec:
    where = f"{entry.registry}/{entry.key}"
    if isinstance(model, type) and issubclass(model, BaseModel):
        aliased = [
            n
            for n, f in model.model_fields.items()
            if f.validation_alias is not None and f.validation_alias != f.alias
        ]
        if aliased:
            raise PluginError(
                f"{where}: params model {entry.params!r} uses validation aliases ({aliased})",
                hint="use alias= (the name the config and the target use) instead",
            )
        names = {f.alias or n for n, f in model.model_fields.items()}
        return _ParamSpec(frozenset(names), model, None)
    if dataclasses.is_dataclass(model) and isinstance(model, type):
        names = {f.name for f in dataclasses.fields(model) if f.init}
        return _ParamSpec(frozenset(names), None, TypeAdapter(model))
    if isinstance(model, type) and issubclass(model, dict) and hasattr(model, "__required_keys__"):
        # any TypedDict (typing or typing_extensions)
        names = {*model.__required_keys__, *getattr(model, "__optional_keys__", ())}
        return _ParamSpec(frozenset(names), None, TypeAdapter(model))
    raise PluginError(
        f"{entry.registry}/{entry.key}: params {entry.params!r} is not a pydantic model, "
        "dataclass or TypedDict",
        hint="point params at a model class, or omit it to validate against the signature",
    )


def _spec_from_signature(entry: Entry, target: Any) -> _ParamSpec:
    try:
        signature = inspect.signature(target, eval_str=True)
    except (NameError, TypeError, SyntaxError):  # annotations that cannot be evaluated
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return _ParamSpec(None, None, None)
    except ValueError:  # builtins without a signature
        return _ParamSpec(None, None, None)
    accepts_any = False
    fields: dict[str, Any] = {}
    names: set[str] = set()
    for index, param in enumerate(signature.parameters.values()):
        if param.kind is param.VAR_KEYWORD:
            accepts_any = True
            continue
        if param.kind in (param.VAR_POSITIONAL, param.POSITIONAL_ONLY):
            continue
        annotation = param.annotation
        if annotation is param.empty or isinstance(annotation, str):
            annotation = Any
        default = ... if param.default is param.empty else param.default
        fields[f"p{index}"] = (annotation, Field(default, alias=param.name))
        names.add(param.name)
    config = ConfigDict(extra="allow" if accepts_any else "forbid", arbitrary_types_allowed=True)
    name = f"{entry.registry}.{entry.key}.params"
    try:
        model = _params_model(name, config, fields)
    except Exception:  # an annotation pydantic cannot handle: check names and presence only
        loose = {
            key: (Any, Field(info.default, alias=info.alias)) for key, (_, info) in fields.items()
        }
        model = _params_model(name, config, loose)
    return _ParamSpec(None if accepts_any else frozenset(names), model, None)


def _params_model(name: str, config: ConfigDict, fields: dict[str, Any]) -> type[BaseModel]:
    return create_model(name, __config__=config, **fields)
