"""The metric spec grammar, ``name@k=v,k=v``: how every layer that names a metric spells it --
``dfwb eval --metrics``, a training config's ``eval.metrics``, aggregation modes, and an adapter
card's ``reported``/``parity`` numbers -- parsed and checked here, once, against the ``metrics``
registry. Pure text and registry lookups, so any layer can use it without numpy or torch.
"""

from __future__ import annotations

from dfwb.core.errors import ConfigError
from dfwb.core.plugins import get_registry

__all__ = ["check_metric_spec", "parse_metric_spec"]


def parse_metric_spec(spec: str) -> tuple[str, dict[str, bool | int | float | str]]:
    """Parse ``name@k=v,k=v`` into ``(name, params)``.

    ``name`` alone (no ``@``) is a metric or aggregation mode with no parameters. Each value is
    read as a bool (``true``/``false``, case-insensitive), else an int, else a float, else left
    as a string; the registry that owns ``name`` then validates and coerces it against that
    target's own parameter types.

    Raises:
        ConfigError: ``spec`` is empty, has no name before ``@``, has ``@`` with nothing after
            it, or a parameter piece is not ``key=value``.
    """
    if not spec.strip():
        raise ConfigError("empty metric spec", hint="use e.g. 'auc' or 'tpr@fpr=0.01'")
    name, sep, rest = spec.partition("@")
    name = name.strip()
    if not name:
        raise ConfigError(f"{spec!r}: missing a name before '@'", hint="use e.g. 'tpr@fpr=0.01'")
    params: dict[str, bool | int | float | str] = {}
    if sep:
        if not rest.strip():
            raise ConfigError(f"{spec!r}: '@' with no parameters", hint="use e.g. 'fpr=0.01'")
        for piece in rest.split(","):
            piece = piece.strip()
            if not piece:
                raise ConfigError(f"{spec!r}: empty parameter", hint="use e.g. 'k=v,k=v'")
            key, eq, value = piece.partition("=")
            key = key.strip()
            if not eq or not key:
                raise ConfigError(
                    f"{spec!r}: parameter {piece!r} is not key=value", hint="use e.g. 'fpr=0.01'"
                )
            if key in params:
                raise ConfigError(
                    f"{spec!r}: parameter {key!r} given twice", hint="pass each parameter once"
                )
            params[key] = _coerce(value.strip())
    return name, params


def _coerce(value: str) -> bool | int | float | str:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def check_metric_spec(spec: str) -> None:
    """Check ``spec`` without computing anything: its syntax, its metric's name and its
    parameters, exactly as :func:`dfwb.eval.metrics.compute` would take them.

    Raises:
        ConfigError: The spec is malformed, or a parameter is unknown, missing or ill-typed.
        UnknownKeyError: No metric has that name.
    """
    name, params = parse_metric_spec(spec)
    get_registry("metrics").validate(name, **params)
