"""The shipped face-processing profiles: named recipes, packaged as plain YAML data.

Each file under the sibling ``profiles/`` directory is one ``ProcessingProfile``, kept as data so
a user's own YAML file with the same shape works exactly the same way. ``load_profile`` resolves
either a shipped name or a path; nothing here needs cv2, PyAV or numpy.
"""

from __future__ import annotations

import importlib.resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from dfwb.core.errors import ContractError, UnknownKeyError, did_you_mean, validation_messages
from dfwb.core.records import ProcessingProfile

__all__ = ["builtin_profiles", "load_profile"]

_PACKAGE = "dfwb.preprocess.face"
_SUFFIXES = (".yaml", ".yml")


def _profiles_dir() -> importlib.resources.abc.Traversable:
    return importlib.resources.files(_PACKAGE) / "profiles"


def builtin_profiles() -> list[str]:
    """The names of every profile shipped with dfwb, sorted."""
    names = [
        resource.name.rsplit(".", 1)[0]
        for resource in _profiles_dir().iterdir()
        if resource.is_file() and resource.name.endswith(_SUFFIXES)
    ]
    return sorted(names)


def _parse(source: str, text: str) -> ProcessingProfile:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"{source}: invalid YAML ({exc})", hint="fix the YAML syntax") from None
    try:
        return ProcessingProfile.model_validate(data)
    except ValidationError as exc:
        raise ContractError(
            f"{source}: " + "; ".join(validation_messages(exc)),
            hint="see `dfwb schema export c3` for the ProcessingProfile shape",
        ) from None


def load_profile(name_or_path: str) -> ProcessingProfile:
    """Load a processing profile by its shipped name, or from a YAML file at a path.

    Raises:
        UnknownKeyError: ``name_or_path`` is neither a shipped profile name nor an existing file.
        ContractError: the YAML is invalid, or it does not validate as a ``ProcessingProfile``.
    """
    built_in = builtin_profiles()
    if name_or_path in built_in:
        resource = _profiles_dir() / f"{name_or_path}.yaml"
        return _parse(name_or_path, resource.read_text("utf-8"))
    path = Path(name_or_path)
    if path.is_file():
        return _parse(str(path), path.read_text("utf-8"))
    hint = (
        f"built-in profiles: {', '.join(built_in)}"
        if built_in
        else "no built-in profiles are shipped"
    )
    raise UnknownKeyError(
        f"no processing profile named {name_or_path!r}{did_you_mean(name_or_path, built_in)}",
        hint=hint,
    )
