"""Safe YAML reading of pack cards (contract C3a): ``yaml.safe_load`` plus pydantic, nothing else.

Every card in a protocol pack is untrusted, third-party data (a plugin can point a pack anywhere
on disk), so this module never does more than parse plain YAML and validate it against a model.
Any failure -- a missing file, invalid YAML, or a card that fails validation -- becomes a single
:class:`ContractError` naming the file, never a raw exception.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from dfwb.core.errors import ContractError, validation_messages
from dfwb.core.records import DatasetCard, LabelVocab

__all__ = ["read_card", "read_labels", "read_model"]


def read_model[M: BaseModel](path: Path, model: type[M]) -> M:
    """Read and validate ``path`` as ``model``. Any failure raises :class:`ContractError`."""
    try:
        text = path.read_text("utf-8")
    except OSError as exc:
        raise ContractError(
            f"{path.name}: cannot read ({type(exc).__name__}: {exc})",
            hint="check that the pack was installed correctly",
        ) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(
            f"{path.name}: invalid YAML ({exc})", hint="fix the YAML syntax"
        ) from None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ContractError(
            f"{path.name}: " + "; ".join(validation_messages(exc)),
            hint=f"not a valid {model.__name__}",
        ) from None


def read_card(dataset_dir: Path) -> DatasetCard:
    """Read ``<dataset_dir>/dataset.yaml``."""
    return read_model(dataset_dir / "dataset.yaml", DatasetCard)


def read_labels(dataset_dir: Path) -> LabelVocab:
    """Read ``<dataset_dir>/labels.yaml``."""
    return read_model(dataset_dir / "labels.yaml", LabelVocab)
