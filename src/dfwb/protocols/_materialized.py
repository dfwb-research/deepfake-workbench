"""Where materialised lists live under the work root, and the command that writes them."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dfwb.protocols.refs import ProtocolRef

__all__ = ["HASHES_FILE", "materialize_command", "materialized_dir"]

#: The file of a materialised recipe dataset that records the card hashes its lists matched.
HASHES_FILE: Final = "hashes.json"


def materialized_dir(work_root: Path, dataset: str) -> Path:
    """``<work root>/<dataset>/materialized``: where ``dfwb protocols materialize`` writes."""
    return work_root / dataset / "materialized"


def materialize_command(ref: ProtocolRef) -> str:
    """The command that materialises every list of ``ref``'s dataset (keeping any pack scope)."""
    scope = f"{ref.pack}:" if ref.pack else ""
    return f"dfwb protocols materialize {scope}{ref.dataset}"
