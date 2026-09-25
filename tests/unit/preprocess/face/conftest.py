"""Shares the ``env`` fixture (built once in ``test_runner.py``) with every test module in this
directory. A conftest re-export makes it an ordinary, pytest-discovered fixture here, rather than
a plain cross-module import that a linter would flag as redefined by every test function's own
``env`` parameter.
"""

from __future__ import annotations

from tests.unit.preprocess.face.test_runner import env

__all__ = ["env"]
