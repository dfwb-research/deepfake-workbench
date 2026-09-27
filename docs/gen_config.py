"""Generate ``reference/config.md`` from the pydantic experiment-config models.

Run by the ``gen-files`` MkDocs plugin at build time; nothing here is committed. ``dfwb config
validate`` checks a config against :class:`dfwb.core.config.schema.TrainConfig`
(``schema: dfwb.train/1``); this page documents that model and every section it is built from, in
one place, generated from the models themselves rather than hand-copied. ``dfwb.core`` is
torch-free by the project's own import-linter contract, so importing this module needs no extra
installed.
"""

from __future__ import annotations

import inspect

import mkdocs_gen_files
from pydantic import BaseModel

from dfwb.core.config import schema as config_schema

_OUT = "reference/config.md"

# TrainConfig, then its sections in the order a config file lists them; everything else the
# module defines (pydantic models only -- not helper functions) follows, alphabetically.
_FEATURED = (
    "TrainConfig",
    "RunSection",
    "DataSection",
    "DataSource",
    "SuiteRef",
    "ClipSection",
    "ClipsPerVideo",
    "TransformsSection",
    "LoaderSection",
    "ModelSection",
    "InputOverrides",
    "TrainSection",
    "EarlyStopSection",
    "EvalSection",
    "ComponentSpec",
)


def _model_classes() -> list[str]:
    found = {
        name: obj
        for name, obj in inspect.getmembers(config_schema, inspect.isclass)
        if obj.__module__ == config_schema.__name__ and issubclass(obj, BaseModel)
    }
    ordered = [name for name in _FEATURED if name in found]
    ordered += sorted(name for name in found if name not in _FEATURED)
    return ordered


def generate() -> None:
    lines = [
        "# Config reference",
        "",
        "Every experiment config (`schema: dfwb.train/1`) validates against "
        "`dfwb.core.config.schema.TrainConfig`; `dfwb config validate` reports every problem "
        "against this shape. This page is generated from the models themselves; the same shape "
        "as a JSON Schema is [C2 in Contracts](contracts/c2.md).",
        "",
        "A pluggable field (`ComponentSpec`, `{name: <key>, **params}`) is checked against the "
        "named registry key's own signature or params model, not against a fixed shape here -- "
        "see [Writing a plugin](../guides/write-a-plugin.md).",
        "",
    ]
    for name in _model_classes():
        lines.append(f"::: dfwb.core.config.schema.{name}")
        lines.append("    options:")
        lines.append("      show_if_no_docstring: true")
        lines.append("      show_signature_annotations: true")
        lines.append("      heading_level: 2")
        lines.append("")
    with mkdocs_gen_files.open(_OUT, "w") as handle:
        handle.write("\n".join(lines))


generate()
