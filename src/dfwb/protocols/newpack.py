"""Scaffolding a new protocol pack distribution (contracts C1, C3a): ``dfwb protocols new-pack``.

:func:`new_pack` writes the small, dependency-free skeleton a pack author starts from: a
``pyproject.toml`` that registers ``register()`` through the ``dfwb.plugins`` entry point, the
loader itself, an empty ``pack.yaml`` (``datasets: []``), and the README/NOTICE/licence text a
human then edits. Every file is rendered from a ``string.Template`` under ``_newpack/`` -- no
Jinja, no other templating dependency -- so the framework's own install stays exactly as
dependency-free as the pack it scaffolds. Nothing here writes media, frames or anything derived
from pixels: an empty pack has none to derive them from.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from string import Template

from dfwb.core.errors import ConfigError

__all__ = ["new_pack"]

# The same shape a registry key must have (lower-kebab-case, C1): this name becomes both the
# distribution name and the key it registers itself under, so it is checked before anything is
# written rather than failing later, inside the generated ``register()``.
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_TEMPLATE_PACKAGE = "dfwb.protocols._newpack"

# (template file under ``_newpack/``, path under the new package -- itself a template,
# substituting ``$import``) -- one output file per template, in the order they are written.
_FILES: tuple[tuple[str, str], ...] = (
    ("pyproject.toml.tmpl", "pyproject.toml"),
    ("README.md.tmpl", "README.md"),
    ("NOTICE.md.tmpl", "NOTICE.md"),
    ("LICENSE-DATA.tmpl", "LICENSE-DATA"),
    ("__init__.py.tmpl", "src/$import/__init__.py"),
    ("pack.yaml.tmpl", "src/$import/packs/pack.yaml"),
)


def _render(template_name: str, mapping: dict[str, str]) -> str:
    text = resources.files(_TEMPLATE_PACKAGE).joinpath(template_name).read_text("utf-8")
    return Template(text).substitute(mapping)


def new_pack(directory: Path, *, name: str, author: str = "Your Name") -> list[Path]:
    """Scaffold a new protocol pack distribution under ``directory``.

    ``directory`` is created if it does not exist; if it does exist, it must be empty. ``name``
    must be lower-kebab-case: it becomes the distribution name, the import name (``-`` -> ``_``)
    and the key the generated ``register()`` publishes itself under. Returns every path written,
    sorted.

    Raises:
        ConfigError: ``directory`` already holds something, or ``name`` is not lower-kebab-case.
    """
    if not _NAME_RE.match(name):
        raise ConfigError(
            f"{name!r} is not a valid pack name",
            hint="pack names are lower-kebab-case, e.g. 'my-protocols'",
        )
    if directory.exists() and any(directory.iterdir()):
        raise ConfigError(
            f"{directory} is not empty",
            hint="run new-pack into an empty or not-yet-created directory",
        )

    mapping = {"name": name, "import": name.replace("-", "_"), "author": author}
    written: list[Path] = []
    for template_name, relative in _FILES:
        target = directory / Template(relative).substitute(mapping)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_render(template_name, mapping), encoding="utf-8")
        written.append(target)
    return sorted(written)
