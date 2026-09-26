"""Code strategies: how an adapter's upstream model code is obtained (``card.code_strategy``).

``pip`` needs nothing here -- the adapter's own extra pins the upstream package, and the adapter
imports it normally. The other two strategies exist because upstream code is not always something
DFWB can simply depend on:

- ``vendored``: upstream code under an MIT/BSD/Apache-compatible licence is copied into
  ``dfwb/zoo/_vendor/<name>/`` verbatim; :func:`check_vendored_layout` checks that copy carries its
  own ``LICENSE``, a non-empty ``NOTICE`` and a hash list of every file it claims to be unmodified
  copies of.
- ``pinned-clone``: upstream code under an incompatible licence (or too large to vendor) is cloned
  at its exact pinned commit into the cache (:func:`ensure_clone`, gated on the card's own licence
  first), never committed to this repository, and imported as a package private to that one
  adapter (:func:`import_pinned_entry`) -- never added to :data:`sys.path`, so a repository laid
  out with its own top-level ``src/`` (or any other name that would otherwise collide with
  something already importable) is loaded under a name of its own instead. An entry file that
  does an absolute import of its own top-level package name (rather than a relative one) usually
  fails outright, since ``sys.path`` never has the clone on it -- unless something *else* of that
  same name is already importable, in which case the entry would otherwise silently bind to that
  instead: :func:`import_pinned_entry` detects and refuses this whenever the colliding name is
  pulled in newly during the call itself (a name already present beforehand is a documented, known
  gap -- see that function's own docstring).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.core.hashing import sha256_file
from dfwb.core.paths import require_root, resolve_roots
from dfwb.zoo.adapter import require_license_accepted
from dfwb.zoo.card import AdapterCard

__all__ = [
    "check_vendored_layout",
    "clone_at_commit",
    "clone_cache_dir",
    "ensure_clone",
    "import_pinned_entry",
    "private_module_name",
]

_LICENSE_FILE = "LICENSE"
_NOTICE_FILE = "NOTICE"
_HASHES_FILE = "HASHES.sha256"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_GIT_TIMEOUT = 300.0


# --------------------------------------------------------------------------------------- vendored


def check_vendored_layout(vendor_dir: Path) -> list[str]:
    """Every problem with ``vendor_dir`` as a vendored copy of upstream code; ``[]`` means it is
    a valid layout. Checks that ``LICENSE`` exists, ``NOTICE`` exists and is non-empty, and the
    hash list itself (``HASHES.sha256``, one ``<sha256>  <relative path>`` line per vendored file)
    exists, names no absolute path or one that escapes the directory, lists at least one file, and
    that every file it names exists and still hashes to what it claims -- catching an edit
    slipped into a copy that is supposed to be verbatim.
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
    elif not notice_path.read_text("utf-8").strip():
        problems.append(f"{_NOTICE_FILE} is empty")
    hashes_path = vendor_dir / _HASHES_FILE
    if not hashes_path.is_file():
        problems.append(f"missing {_HASHES_FILE}")
        return problems
    listed = 0
    for lineno, line in enumerate(hashes_path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            problems.append(f"{_HASHES_FILE}:{lineno}: malformed line {line!r}")
            continue
        expected_sha256, relative = parts
        pure = Path(relative)
        if pure.is_absolute() or ".." in pure.parts:
            problems.append(
                f"{_HASHES_FILE}:{lineno}: {relative!r} is absolute or escapes the vendor directory"
            )
            continue
        listed += 1
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
    if listed == 0:
        problems.append(f"{_HASHES_FILE} lists no files")
    return problems


# ------------------------------------------------------------------------------------ pinned-clone


def _validate_commit(commit: str) -> None:
    if not _COMMIT_RE.fullmatch(commit):
        raise ConfigError(
            f"pinned-clone: {commit!r} is not a full 40-character commit sha",
            hint="pin an exact commit sha (never a branch, tag or short sha)",
        )


def _git_env() -> dict[str, str]:
    # Never wait on a credential prompt (HTTPS) or a host-key/passphrase prompt (SSH): a private
    # or mistyped repo url should fail fast, not hang the whole scoring run.
    return {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes",
    }


def _run_git(args: list[str], *, timeout: float = _GIT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=_git_env(),
        )
    except FileNotFoundError:
        raise InstallationError(
            "pinned-clone: git is not installed",
            hint="install git to use the pinned-clone strategy",
        ) from None
    except subprocess.TimeoutExpired:
        raise InstallationError(
            f"pinned-clone: `git {' '.join(args)}` timed out after {timeout:g}s",
            hint="check the repo URL and network access, or that the host is reachable",
        ) from None


def clone_at_commit(repo: str, commit: str, dest: Path) -> Path:
    """Clone ``repo`` into ``dest`` and check out ``commit`` exactly -- never a location a caller
    might later add to :data:`sys.path`.

    Raises:
        ConfigError: ``commit`` is not a full 40-character (lowercase hex) commit sha -- a branch,
            a tag and a short sha are all refused, since none of them pin an exact commit.
        InstallationError: git is not installed, the clone itself failed (bad repo, no network),
            or a git command timed out.
        ContractError: ``commit`` does not exist in ``repo`` (the adapter card's pin is wrong), or
            the checked-out ``HEAD`` does not match it (this should never happen after a
            successful checkout; it is checked anyway).
    """
    _validate_commit(commit)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clone = _run_git(["clone", "--no-checkout", "--", repo, str(dest)])
    if clone.returncode != 0:
        raise InstallationError(
            f"pinned-clone: {repo}: `git clone` failed: {clone.stderr.strip()}",
            hint="check the upstream repo URL and network access",
        )
    checkout = _run_git(["-C", str(dest), "checkout", commit])
    if checkout.returncode != 0:
        raise ContractError(
            f"pinned-clone: {repo}: commit {commit!r} does not exist",
            hint="the adapter card's pinned commit is wrong, or the upstream history was rewritten",
        )
    head = _run_git(["-C", str(dest), "rev-parse", "HEAD"])
    if head.returncode != 0 or head.stdout.strip().lower() != commit:
        raise ContractError(
            f"pinned-clone: {repo}: checked out HEAD {head.stdout.strip()!r} does not match the "
            f"pinned commit {commit!r}",
            hint="this should not happen after a successful checkout; report it as a bug",
        )
    _exclude_pycache(dest)
    return dest


def _exclude_pycache(clone_dir: Path) -> None:
    """Append ``__pycache__/`` and ``*.pyc`` to the clone's own (local-only, never committed)
    ``.git/info/exclude``, so importing an entry from it -- which writes bytecode cache files into
    the tree -- never shows up in ``git status --porcelain``, and so never makes
    :func:`ensure_clone`'s own dirty-worktree check refuse an otherwise perfectly good clone just
    because it has already been used once."""
    exclude_file = clone_dir / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True, exist_ok=True)
    with exclude_file.open("a", encoding="utf-8") as handle:
        handle.write("\n__pycache__/\n*.pyc\n")


def clone_cache_dir(name: str, commit: str) -> Path:
    """``$DFWB_CACHE_ROOT/zoo/<name>/code/<commit>/``."""
    roots = resolve_roots()
    cache_root = require_root("cache", roots)
    return cache_root / "zoo" / name / "code" / commit


def _is_clean_worktree(repo_dir: Path) -> bool:
    status = _run_git(["-C", str(repo_dir), "status", "--porcelain"])
    return status.returncode == 0 and status.stdout.strip() == ""


def _head_commit(repo_dir: Path) -> str | None:
    head = _run_git(["-C", str(repo_dir), "rev-parse", "HEAD"])
    return head.stdout.strip().lower() if head.returncode == 0 else None


def _reuse_existing_clone(name: str, commit: str, dest: Path) -> Path:
    """``dest``, if it is a clean clone still at ``commit``; otherwise raises, naming ``dest`` and
    how to clear it."""
    head = _head_commit(dest)
    if head != commit:
        raise ContractError(
            f"zoo:{name}: {dest} is at commit {head!r}, not the pinned {commit!r}",
            hint=f"remove {dest} and let it clone again",
        )
    if not _is_clean_worktree(dest):
        raise ContractError(
            f"zoo:{name}: {dest} has local modifications",
            hint=f"remove {dest} and let it clone again",
        )
    return dest


def ensure_clone(card: AdapterCard) -> Path:
    """The local pinned clone of ``card.upstream``, cloning it once if needed.

    Order: the licence gate first (:func:`dfwb.zoo.adapter.require_license_accepted`), then the
    clone itself, into ``$DFWB_CACHE_ROOT/zoo/<name>/code/<commit>/`` -- built in a temporary
    sibling directory and renamed into place only once it fully succeeds, so a failed clone leaves
    nothing behind (a retry starts clean, not from a half-cloned directory). An existing clone is
    reused only when its own ``HEAD`` still matches the pin and it has no local modifications;
    otherwise this refuses to guess and raises instead. If another process finishes cloning the
    same pin first (the rename into place then fails because ``dest`` now exists), the temporary
    clone is discarded and the winner's clone is reused instead, under the same checks.

    Raises:
        ConfigError: ``card`` has no ``upstream`` block, or its commit is not a full 40-character
            sha.
        InstallationError: the licence must be acknowledged first, or git is not installed, the
            clone failed, or it timed out.
        ContractError: a clone already at the cache path (whether found there at the start, or
            left by a process that won the race to clone it) is at a different commit, or has
            local modifications -- either way, names the path and how to clear it.
    """
    if card.upstream is None:
        raise ConfigError(
            f"zoo:{card.name}: pinned-clone needs an `upstream` block in the card",
            hint="add `upstream: {repo: ..., commit: ...}` to the card",
        )
    require_license_accepted(card)
    upstream = card.upstream

    dest = clone_cache_dir(card.name, upstream.commit)
    if dest.is_dir():
        return _reuse_existing_clone(card.name, upstream.commit, dest)

    tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        clone_at_commit(upstream.repo, upstream.commit, tmp)
        try:
            tmp.replace(dest)
        except OSError:
            # Lost a race with another process that finished cloning the same pin first.
            return _reuse_existing_clone(card.name, upstream.commit, dest)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return dest


def private_module_name(name: str) -> str:
    """``dfwb_zoo_ext_<name>``, the private module name a pinned clone of adapter ``name`` is
    always imported under -- sanitised, so an adapter name can never smuggle in a dotted or
    otherwise invalid module path."""
    safe = re.sub(r"[^a-z0-9_]", "_", name.lower())
    return f"dfwb_zoo_ext_{safe}"


def _register_private_package(clone_root: Path, module_name: str) -> ModuleType:
    """Register ``module_name`` in :data:`sys.modules` as a package rooted at ``clone_root``,
    importable only under that private name -- ``clone_root`` is never added to :data:`sys.path`,
    so nothing outside an explicit ``import <module_name>...`` can ever see it.

    A registration already present is reused as-is, *unless* it points at a different clone root
    (the same adapter re-cloned at a new pinned commit, say): that stale registration, and
    everything already cached under it, is dropped first, so this never silently keeps serving
    code from a directory that is no longer the adapter's current clone.
    """
    existing = sys.modules.get(module_name)
    if existing is not None:
        if list(getattr(existing, "__path__", None) or []) == [str(clone_root)]:
            return existing
        _purge_module_tree(module_name)
    init_file = clone_root / "__init__.py"
    if init_file.is_file():
        spec = importlib.util.spec_from_file_location(
            module_name, init_file, submodule_search_locations=[str(clone_root)]
        )
    else:
        spec = importlib.util.spec_from_loader(module_name, loader=None, is_package=True)
    if spec is None:
        raise ContractError(
            f"pinned-clone: {clone_root}: cannot be loaded as a Python package",
            hint="check the adapter's pinned clone",
        )
    spec.submodule_search_locations = [str(clone_root)]
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if spec.loader is not None:
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ContractError(
                f"pinned-clone: {clone_root}: raised {type(exc).__name__}: {exc} while loading "
                "its own __init__.py",
                hint="this is a bug in the adapter's entry file or its pinned commit",
            ) from exc
    return module


def _purge_module_tree(module_name: str) -> None:
    """Remove ``module_name`` and everything registered under it from :data:`sys.modules`."""
    prefix = f"{module_name}."
    for key in list(sys.modules):
        if key == module_name or key.startswith(prefix):
            sys.modules.pop(key, None)


def _is_importable_at_clone_root(clone_root: Path, name: str) -> bool:
    """Whether ``name`` names something *actually importable* at ``clone_root`` -- a subdirectory
    with its own ``__init__.py``, or a ``<name>.py`` file -- not just any subdirectory of that
    name. A plain data folder (``datasets/``, say, with no ``__init__.py``) is never itself
    importable, so a name that merely happens to match one is not a false positive for
    :func:`_find_leaked_top_level_names`."""
    package_dir = clone_root / name
    if package_dir.is_dir() and (package_dir / "__init__.py").is_file():
        return True
    return (clone_root / f"{name}.py").is_file()


def _find_leaked_top_level_names(clone_root: Path, before: frozenset[str]) -> list[str]:
    """Every top-level name that (a) newly appeared in :data:`sys.modules` since ``before`` and
    (b) is also actually importable at ``clone_root`` -- the clone's own real name, resolved
    instead to whatever else happens to be importable under that same name elsewhere on
    :data:`sys.path` (never the clone's own code, since ``clone_root`` is never added to it).

    This is the one thing importing a pinned entry cannot make safe entirely on its own: an entry
    file that imports its own top-level package name *absolutely*, rather than relatively,
    silently binds to the wrong module instead of failing outright. Only a name that is genuinely
    new *during this call* can be caught this way -- one already present in :data:`sys.modules`
    from earlier is invisible to the comparison and binds silently; see
    :func:`import_pinned_entry`.
    """
    return sorted(
        name
        for name in sys.modules
        if name not in before and "." not in name and _is_importable_at_clone_root(clone_root, name)
    )


def _purge_after_failed_import(module_name: str, leaked: Sequence[str]) -> None:
    """Remove everything a failed or leaky pinned-entry import could have left inconsistent:
    every submodule already registered under the private root package -- which makes a refusal
    stick, since the entry's own (bad) module is one such submodule, so a retry never silently
    returns it straight out of :data:`sys.modules` without being re-attempted -- each leaked
    top-level name, and any of *their* own submodules (a leaked ``src`` takes ``src.helper`` with
    it). The private root package itself (``module_name``, with no submodules) is left registered:
    it may still be perfectly good, and re-running its own ``__init__.py`` on every retry would be
    wasteful.
    """
    leaked_set = set(leaked)
    prefixes = [f"{module_name}."] + [f"{name}." for name in leaked]
    for key in list(sys.modules):
        if key in leaked_set or any(key.startswith(prefix) for prefix in prefixes):
            sys.modules.pop(key, None)


def _leak_error(clone_root: Path, leaked: Sequence[str]) -> ContractError:
    return ContractError(
        f"pinned-clone: {clone_root}: importing it pulled in {list(leaked)!r} from elsewhere on "
        "sys.path, not from the clone itself",
        hint="the upstream code imports its own top-level package name absolutely; vendor a "
        "small shim instead (the vendored code strategy), or ask upstream to use relative "
        "imports",
    )


def import_pinned_entry(clone_root: Path, card: AdapterCard, entry: str) -> ModuleType:
    """Import ``<entry>`` from the pinned clone at ``clone_root``, registered privately under
    ``dfwb_zoo_ext_<card.name>`` (:func:`private_module_name`, always derived from the card itself
    -- never a name a caller supplies) -- never added to :data:`sys.path`. Because that private
    name is a real package rooted at ``clone_root``, a *relative* import inside the entry
    (``from . import sibling``) resolves correctly.

    An *absolute* import of the clone's own top-level package name does not resolve there, and
    :data:`sys.modules` is compared before and after to catch it: on either path (the import
    itself raising, or completing without error), a top-level name that is both new since this
    call started and actually importable at ``clone_root`` (see
    :func:`_is_importable_at_clone_root`) is refused rather than silently kept, and the refusal
    sticks -- everything it and the private package's own submodules could have left behind is
    removed, so a second call re-attempts the import rather than returning stale state. **Known
    limitation:** a colliding name already present in :data:`sys.modules` *before* this call is
    invisible to the comparison and binds silently; only a name pulled in newly during this
    specific call can be caught this way. No custom import machinery is installed to close that
    gap; vendoring is the documented way around an upstream that does this.

    Raises:
        ContractError: the entry cannot be imported, raises while importing, or (on either path)
            its own absolute self-imports resolved to some other, already-importable package of
            the same name -- in which case the message names what leaked and hints at vendoring,
            not the generic "bug in the adapter" wording a plain import failure gets.
    """
    module_name = private_module_name(card.name)
    _register_private_package(clone_root, module_name)
    before = frozenset(sys.modules)
    full_name = f"{module_name}.{entry}"

    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.import_module(full_name)
    except Exception as exc:
        leaked = _find_leaked_top_level_names(clone_root, before)
        _purge_after_failed_import(module_name, leaked)
        if leaked:
            raise _leak_error(clone_root, leaked) from exc
        raise ContractError(
            f"pinned-clone: {full_name}: raised {type(exc).__name__}: {exc} while importing",
            hint="this is a bug in the adapter's entry module or its pinned commit",
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

    leaked = _find_leaked_top_level_names(clone_root, before)
    if leaked:
        _purge_after_failed_import(module_name, leaked)
        raise _leak_error(clone_root, leaked)
    return module
