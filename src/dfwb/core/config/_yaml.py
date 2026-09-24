"""YAML for configs: safe loading, with YAML 1.2 exponent floats.

PyYAML follows YAML 1.1, which reads ``1e-4`` and ``3e-4`` (no dot) as strings. Configs are full
of learning rates written that way, and a string there would change the config fingerprint and
reach the optimiser as text, so configs are read with :class:`ConfigLoader` instead.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

__all__ = ["ConfigLoader", "load_yaml"]


class ConfigLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that also reads ``1e-4`` / ``3E+5`` as floats."""


ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:[0-9][0-9_]*(?:\.[0-9_]*)?|\.[0-9_]+)[eE][-+]?[0-9]+$"),
    list("-+0123456789."),
)


def load_yaml(text: str) -> Any:
    """Parse ``text`` with :class:`ConfigLoader` (safe: plain data only, never objects)."""
    return yaml.load(text, Loader=ConfigLoader)
