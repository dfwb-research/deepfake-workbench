"""Protocol reference grammar (contract C3a): ``[<pack>:]<dataset>[/<scheme>][@<pin>]``.

References name a dataset (optionally scoped to a pack, when the plain dataset id is ambiguous
across installed packs), an optional scheme within it, and an optional pin: either a pack version
(``@1.2.0``) or a scheme hash prefix (``@3f9a1c2e``), so score files and configs can record exactly
what they ran against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dfwb.core.errors import ConfigError

__all__ = ["ProtocolRef", "parse_ref"]

_KEBAB = r"[a-z0-9]+(?:-[a-z0-9]+)*"
_SCHEME = r"[a-z0-9][a-z0-9+.-]*"
_PIN = r"\d+\.\d+\.\d+|[0-9a-f]{6,64}"
_GRAMMAR = "[<pack>:]<dataset>[/<scheme>][@<version or hash>]"

_REF = re.compile(
    rf"^(?:(?P<pack>{_KEBAB}):)?(?P<dataset>{_KEBAB})"
    rf"(?:/(?P<scheme>{_SCHEME}))?(?:@(?P<pin>{_PIN}))?$"
)


@dataclass(frozen=True)
class ProtocolRef:
    """A parsed protocol reference."""

    pack: str | None
    dataset: str
    scheme: str | None
    pin: str | None


def parse_ref(text: str) -> ProtocolRef:
    """Parse a protocol reference, e.g. ``"dfwb-protocols:ffpp/official@1.2.0"``.

    Raises:
        ConfigError: ``text`` does not match the grammar.
    """
    match = _REF.match(text)
    if match is None:
        raise ConfigError(
            f"invalid protocol reference {text!r}",
            hint=f"protocol references look like {_GRAMMAR}, e.g. 'ffpp/official@1.2.0'",
        )
    return ProtocolRef(match["pack"], match["dataset"], match["scheme"], match["pin"])
