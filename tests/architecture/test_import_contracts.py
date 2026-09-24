"""import-linter enforces the layer table and the forbidden imports (see [tool.importlinter])."""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_import_linter_contracts_are_kept():
    lint = Path(sys.executable).parent / "lint-imports"
    done = subprocess.run(
        [str(lint), "--no-cache"], cwd=REPO, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stdout + done.stderr
