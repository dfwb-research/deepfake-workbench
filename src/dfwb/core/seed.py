"""Seeding. Libraries are seeded only if they are installed; nothing is imported otherwise."""

from __future__ import annotations

import importlib
import importlib.util
import os
import random

__all__ = ["seed_everything"]


def _installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python's ``random``, NumPy and PyTorch (when installed).

    Args:
        seed: The seed.
        deterministic: Also ask PyTorch for deterministic algorithms (slower; errors on
            operations that have no deterministic implementation).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects child processes only
    if _installed("numpy"):
        numpy = importlib.import_module("numpy")
        numpy.random.seed(seed)
    if _installed("torch"):
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)  # seeds every device
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
