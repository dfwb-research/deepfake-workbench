"""Code strategies: how an adapter's upstream model code is obtained (``card.code_strategy``).

``pip`` needs nothing here -- the adapter's own extra pins the upstream package, and the adapter
imports it normally. The other two strategies exist because upstream code is not always something
DFWB can simply depend on:

- ``vendored``: upstream code under an MIT/BSD/Apache-compatible licence is copied into
  ``dfwb/zoo/_vendor/<name>/`` verbatim; :func:`check_vendored_layout` checks that copy carries its
  own ``LICENSE``, a ``NOTICE`` and a hash list of every file it claims to be unmodified copies of.
- ``pinned-clone``: upstream code under an incompatible licence (or too large to vendor) is cloned
  at its pinned commit into the cache, never committed to this repository, and imported under a
  private module name via :func:`load_pinned_module` -- never added to ``sys.path`` as a top-level
  package, so a repository laid out with its own top-level ``src/`` (or any other name that would
  otherwise collide with something already importable) never pollutes :data:`sys.modules`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from dfwb.core.errors import ContractError, InstallationError
from dfwb.core.hashing import sha256_file

__all__ = [
    "check_vendored_layout",
    "clone_at_commit",
    "load_pinned_module",
]

_LICENSE_FILE = "LICENSE"
_NOTICE_FILE = "NOTICE"
_HASHES_FILE = "HASHES.sha256"


# --------------------------------------------------------------------------------------- vendored


def check_vendored_layout(vendor_dir: Path) -> list[str]:
    """Every problem with ``vendor_dir`` as a vendored copy of upstream code; ``[]`` means it is
    a valid layout. Checks that ``LICENSE``, ``NOTICE`` and the hash list itself
    (``HASHES.sha256``, one ``<sha256>  <relative path>`` line per vendored file) all exist, and
    that every file the hash list names exists and still hashes to what it claims -- catching an
    edit slipped into a copy that is supposed to be verbatim.
    """
    problems: list[str] = []
    if not vendor_dir.is_dir():
        return [f"{vendor_dir}: no such directory"]
    license_path = vendor_dir / _LICENSE_FILE
    if not license_path.is_file():
        problems.append(f"missing {_LICENSE_FILE}")
    notice_path = vendor_dir / _NOTICE_FILE
    if not notice_path.is_file():
        problems.append(f"missing {_NOTICE_FILE}")
    hashes_path = vendor_dir / _HASHES_FILE
    if not hashes_path.is_file():
        problems.append(f"missing {_HASHES_FILE}")
        return problems
    for lineno, line in enumerate(hashes_path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"{_HASHES_FILE}:{lineno}: malformed line {line!r}")
            continue
        expected_sha256, relative = parts
        target = vendor_dir / relative
        if not target.is_file():
            problems.append(f"{_HASHES_FILE}:{lineno}: {relative} does not exist")
            continue
        actual_sha256 = sha256_file(target)
        if actual_sha256 != expected_sha256:
            problems.append(
                f"{_HASHES_FILE}:{lineno}: {relative} hashes to {actual_sha256}, "
                f"expected {expected_sha256} (the file has changed since it was vendored)"
            )
    return problems


# ------------------------------------------------------------------------------------ pinned-clone


def _run_git(args: list[str], *, where: str) -> None:
    try:
        result = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=300
        )
    except FileNotFoundError:
        raise InstallationError(
            f"{where}: git is not installed", hint="install git to use the pinned-clone strategy"
        ) from None
    if result.returncode != 0:
        raise InstallationError(
            f"{where}: `git {' '.join(args)}` failed: {result.stderr.strip()}",
            hint="check the upstream repo URL and network access",
        )


def clone_at_commit(repo: str, commit: str, dest: Path) -> Path:
    """Clone ``repo`` into ``dest`` and check out ``commit`` exactly, into the cache -- never a
    location a caller might later add to ``sys.path``.

    Raises:
        InstallationError: git is not installed, or the clone itself failed (bad repo, no
            network).
        ContractError: ``commit`` does not exist in ``repo`` (the adapter card's pin is wrong).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", "--no-checkout", repo, str(dest)], where=f"pinned-clone: {repo}")
    try:
        subprocess.run(
            ["git", "-C", str(dest), "checkout", commit],
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
    except subprocess.CalledProcessError as exc:
        raise ContractError(
            f"pinned-clone: {repo}: commit {commit!r} does not exist",
            hint="the adapter card's pinned commit is wrong, or the upstream history was rewritten",
        ) from exc
    return dest


def load_pinned_module(entry_file: Path, module_name: str) -> ModuleType:
    """Load ``entry_file`` as ``module_name``, without ever adding its directory to
    :data:`sys.path`: only ``module_name`` itself is ever added to :data:`sys.modules`, so a
    top-level package name the cloned repository happens to use (``src``, most often) never
    shadows or collides with anything already imported.

    Raises:
        ContractError: ``entry_file`` cannot be loaded as a module, or raises while executing.
    """
    spec = importlib.util.spec_from_file_location(module_name, entry_file)
    if spec is None or spec.loader is None:
        raise ContractError(
            f"pinned-clone: {entry_file}: cannot be loaded as a Python module",
            hint="check the adapter's entry file path",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ContractError(
            f"pinned-clone: {entry_file}: raised {type(exc).__name__}: {exc} while loading",
            hint="this is a bug in the adapter's entry file or its pinned commit",
        ) from exc
    return module
