"""Code strategies: the ``vendored`` layout checker, and ``pinned-clone`` against a local git
fixture repository (never the real network) -- cloned at its exact commit, cached, and imported as
a package private to the adapter, never touching ``sys.path``."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

import dfwb.zoo.strategies as strategies_module
from dfwb.core import licenses
from dfwb.core.errors import ConfigError, ContractError, InstallationError
from dfwb.zoo.card import AdapterCard, parse_card
from dfwb.zoo.strategies import (
    check_vendored_layout,
    clone_at_commit,
    clone_cache_dir,
    ensure_clone,
    import_pinned_entry,
    private_module_name,
)

# --------------------------------------------------------------------------------------- vendored


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_valid_vendor_dir(root: Path) -> Path:
    vendor = root / "_vendor" / "gend"
    vendor.mkdir(parents=True)
    (vendor / "LICENSE").write_text("MIT\n")
    (vendor / "NOTICE").write_text("vendored from upstream at commit abc\n")
    model_text = "class Model:\n    pass\n"
    (vendor / "model.py").write_text(model_text)
    (vendor / "HASHES.sha256").write_text(f"{_sha(model_text)}  model.py\n")
    return vendor


def test_a_valid_vendored_layout_has_no_problems(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    assert check_vendored_layout(vendor) == []


def test_a_missing_vendor_directory_is_one_problem(tmp_path):
    problems = check_vendored_layout(tmp_path / "nope")
    assert len(problems) == 1
    assert "no such directory" in problems[0]


def test_missing_license_and_notice_are_both_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "LICENSE").unlink()
    (vendor / "NOTICE").unlink()

    problems = check_vendored_layout(vendor)

    assert any("LICENSE" in p for p in problems)
    assert any("NOTICE" in p for p in problems)


def test_missing_hash_list_is_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "HASHES.sha256").unlink()

    problems = check_vendored_layout(vendor)

    assert any("HASHES.sha256" in p for p in problems)


def test_a_modified_file_fails_its_recorded_hash(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "model.py").write_text("class Model:\n    pass  # edited\n")

    problems = check_vendored_layout(vendor)

    assert any("model.py" in p and "expected" in p for p in problems)


def test_blank_lines_in_the_hash_list_are_skipped(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    model_text = (vendor / "model.py").read_text()
    (vendor / "HASHES.sha256").write_text(f"\n{_sha(model_text)}  model.py\n\n")

    assert check_vendored_layout(vendor) == []


def test_a_malformed_hash_list_line_is_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "HASHES.sha256").write_text("not-a-valid-line-with-no-path\n")

    problems = check_vendored_layout(vendor)

    assert any("malformed line" in p for p in problems)


def test_a_hash_list_entry_naming_a_missing_file_is_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "HASHES.sha256").write_text(f"{_sha('x')}  missing.py\n")

    problems = check_vendored_layout(vendor)

    assert any("missing.py" in p and "does not exist" in p for p in problems)


def test_an_empty_notice_is_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "NOTICE").write_text("   \n")

    problems = check_vendored_layout(vendor)

    assert any("NOTICE" in p and "empty" in p for p in problems)


def test_an_empty_hash_list_is_reported(tmp_path):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "HASHES.sha256").write_text("\n")

    problems = check_vendored_layout(vendor)

    assert any("lists no files" in p for p in problems)


@pytest.mark.parametrize(
    "relative",
    ["/etc/passwd", "../../etc/passwd", "sub/../../escape.py"],
    ids=["absolute", "dotdot", "escapes"],
)
def test_a_hash_list_path_that_is_absolute_or_escapes_the_directory_is_reported(tmp_path, relative):
    vendor = _write_valid_vendor_dir(tmp_path)
    (vendor / "HASHES.sha256").write_text(f"{_sha('x')}  {relative}\n")

    problems = check_vendored_layout(vendor)

    assert any("escapes" in p or "absolute" in p for p in problems)


def test_every_shipped_vendored_copy_has_a_valid_layout():
    # No third-party code is vendored yet -- this loop is vacuous today, and stays ready for the
    # first one.
    import dfwb.zoo

    vendor_root = Path(dfwb.zoo.__file__).resolve().parent / "_vendor"
    if not vendor_root.is_dir():
        return
    for entry in sorted(vendor_root.iterdir()):
        if entry.is_dir():
            assert check_vendored_layout(entry) == [], entry


# ------------------------------------------------------------------------------------ pinned-clone


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _rev_parse(cwd: Path, ref: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", ref], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def upstream_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A local git repository with a top-level ``src/`` package (the case that used to force
    vendoring code just to dodge a ``sys.modules`` collision), one file that imports a sibling
    *relatively* (``src/model.py``) and two, at the repo's own top level, that import the clone's
    top-level package name *absolutely* -- ``absolute_entry.py`` (``import src``, succeeds against
    a decoy) and ``absolute_from_helper_entry.py`` (``from src.helper import ...``, which raises
    against a plain-module decoy but succeeds against a decoy *package*) -- plus a branch and a
    tag on the same commit, for the pin-validation tests. Returns ``(repo, commit, branch)``."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.org"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "helper.py").write_text("HELPER_VALUE = 100\n")
    (repo / "src" / "model.py").write_text(
        "from __future__ import annotations\n\n"
        "from .helper import HELPER_VALUE\n\n"
        "VALUE = 42\n\n\n"
        "def make_detector():\n    return VALUE + HELPER_VALUE\n"
    )
    (repo / "absolute_entry.py").write_text(
        "from __future__ import annotations\n\n"
        "import src\n\n"
        "MARKER = getattr(src, 'MARKER', 'clones-own-src')\n"
    )
    (repo / "absolute_from_helper_entry.py").write_text(
        "from __future__ import annotations\n\n"
        "from src.helper import HELPER_VALUE\n\n"
        "MARKER = HELPER_VALUE\n"
    )
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "initial"], repo)
    commit = _rev_parse(repo)
    _git(["tag", "v1.0"], repo)
    branch = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit, branch


def test_clone_at_commit_checks_out_exactly_that_commit(tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    dest = tmp_path / "clone"

    result = clone_at_commit(str(repo), commit, dest)

    assert result == dest
    assert (dest / "src" / "model.py").is_file()
    assert _rev_parse(dest) == commit


def test_clone_at_commit_rejects_a_nonexistent_commit(tmp_path, upstream_repo):
    repo, _, _ = upstream_repo
    with pytest.raises(ContractError, match="does not exist"):
        clone_at_commit(str(repo), "f" * 40, tmp_path / "clone")


def test_clone_at_commit_rejects_a_nonexistent_repo(tmp_path):
    with pytest.raises(InstallationError):
        clone_at_commit(str(tmp_path / "no-such-repo"), "f" * 40, tmp_path / "clone")


def test_clone_at_commit_reports_a_missing_git_binary(tmp_path, upstream_repo, monkeypatch):
    repo, commit, _ = upstream_repo
    monkeypatch.setenv("PATH", "")  # git cannot be found on an empty PATH
    with pytest.raises(InstallationError, match="git is not installed"):
        clone_at_commit(str(repo), commit, tmp_path / "clone")


@pytest.mark.parametrize("kind", ["branch", "tag", "short-sha"])
def test_clone_at_commit_refuses_anything_that_is_not_a_full_pinned_sha(
    tmp_path, upstream_repo, kind
):
    repo, commit, branch = upstream_repo
    ref = {"branch": branch, "tag": "v1.0", "short-sha": commit[:12]}[kind]
    with pytest.raises(ConfigError, match="commit sha"):
        clone_at_commit(str(repo), ref, tmp_path / "clone")


def test_clone_at_commit_disables_the_git_terminal_prompt(tmp_path, upstream_repo, monkeypatch):
    repo, commit, _ = upstream_repo
    seen_envs: list[dict[str, str]] = []
    real_run = strategies_module.subprocess.run

    def spy(args, **kwargs):
        seen_envs.append(kwargs.get("env") or {})
        return real_run(args, **kwargs)

    monkeypatch.setattr(strategies_module.subprocess, "run", spy)
    clone_at_commit(str(repo), commit, tmp_path / "clone")

    assert seen_envs  # at least the clone call was seen
    assert all(env.get("GIT_TERMINAL_PROMPT") == "0" for env in seen_envs)
    assert all(env.get("GIT_SSH_COMMAND") == "ssh -o BatchMode=yes" for env in seen_envs)


def test_clone_at_commit_reports_a_timeout(tmp_path, upstream_repo, monkeypatch):
    repo, commit, _ = upstream_repo

    def timed_out(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(strategies_module.subprocess, "run", timed_out)
    with pytest.raises(InstallationError, match="timed out"):
        clone_at_commit(str(repo), commit, tmp_path / "clone")


def test_clone_at_commit_reports_a_head_that_does_not_match_the_pin(
    tmp_path, upstream_repo, monkeypatch
):
    repo, commit, _ = upstream_repo
    real_run = strategies_module.subprocess.run

    def lying_rev_parse(args, **kwargs):
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout="f" * 40 + "\n", stderr="")
        return real_run(args, **kwargs)

    monkeypatch.setattr(strategies_module.subprocess, "run", lying_rev_parse)
    with pytest.raises(ContractError, match="does not match the pinned commit"):
        clone_at_commit(str(repo), commit, tmp_path / "clone")


# ---------------------------------------------------------------------------------- ensure_clone


def _gated_card_yaml(name: str, repo: Path, commit: str, *, requires_ack: bool = False) -> str:
    return f"""
name: {name}
display_name: Pinned Test
contract_version: [1, 0]
upstream: {{repo: "{repo}", commit: "{commit}"}}
license: {{code: MIT, requires_ack: {"true" if requires_ack else "false"}}}
code_strategy: pinned-clone
input: {{}}
"""


def test_ensure_clone_clones_once(isolated, tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))

    dest = ensure_clone(card)

    assert dest == clone_cache_dir(card.name, commit)
    assert (dest / "src" / "model.py").is_file()


def test_ensure_clone_reuses_an_existing_clean_clone(
    isolated, tmp_path, upstream_repo, monkeypatch
):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    first = ensure_clone(card)

    def boom(*args, **kwargs):
        raise AssertionError("clone_at_commit must not be called on a cache hit")

    monkeypatch.setattr(strategies_module, "clone_at_commit", boom)
    second = ensure_clone(card)

    assert second == first


def test_ensure_clone_refuses_a_dirty_clone(isolated, upstream_repo):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = ensure_clone(card)
    (dest / "src" / "model.py").write_text("VALUE = 999\n")

    with pytest.raises(ContractError, match="local modifications"):
        ensure_clone(card)


def test_ensure_clone_reuses_a_clone_that_has_already_been_imported_from(isolated, upstream_repo):
    """The literal end-to-end scenario: clone, import an entry from it, then ensure_clone again --
    must reuse it, not refuse it as dirty."""
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = ensure_clone(card)
    module_name = private_module_name(card.name)

    try:
        module = import_pinned_entry(dest, card, "src.model")
        assert module.make_detector() == 142

        reused = ensure_clone(card)
        assert reused == dest
    finally:
        _forget(module_name)


def test_ensure_clone_reuses_a_clone_with_bytecode_cache_files(isolated, upstream_repo):
    """Simulates what an import leaves behind (``__pycache__/*.pyc``) even in a scenario where
    something bypasses ``sys.dont_write_bytecode`` (a subprocess spawned by the adapter's own
    code, say): the clone's own ``.git/info/exclude`` must keep those out of
    ``git status --porcelain`` on its own, independent of ``import_pinned_entry`` also having set
    ``sys.dont_write_bytecode`` -- or ``ensure_clone``'s dirty-worktree check would refuse an
    otherwise perfectly good, already-used clone."""
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = ensure_clone(card)

    pycache = dest / "src" / "__pycache__"
    pycache.mkdir()
    (pycache / "model.cpython-312.pyc").write_bytes(b"\x00\x00\x00\x00")
    (dest / "src" / "model.pyc").write_bytes(b"\x00\x00\x00\x00")

    reused = ensure_clone(card)
    assert reused == dest


def test_ensure_clone_refuses_a_clone_at_another_commit(isolated, upstream_repo):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = ensure_clone(card)
    (dest / "extra.txt").write_text("x")
    _git(["add", "."], dest)
    _git(["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "drift"], dest)

    with pytest.raises(ContractError, match="not the pinned"):
        ensure_clone(card)


def test_ensure_clone_retries_after_a_failed_clone_leaves_nothing_behind(isolated, tmp_path):
    bad_card = parse_card(_gated_card_yaml("pinned-test", tmp_path / "no-such-repo", "f" * 40))
    with pytest.raises((InstallationError, ContractError)):
        ensure_clone(bad_card)
    assert not clone_cache_dir("pinned-test", "f" * 40).exists()


def test_ensure_clone_retries_after_a_failed_clone_then_succeeds(isolated, tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    bad_card = parse_card(_gated_card_yaml("pinned-test", tmp_path / "no-such-repo", commit))
    with pytest.raises(InstallationError):
        ensure_clone(bad_card)

    good_card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = ensure_clone(good_card)
    assert (dest / "src" / "model.py").is_file()


def test_ensure_clone_cleans_up_a_partial_clone_left_by_a_bad_commit(isolated, upstream_repo):
    # Unlike a bad repo url (nothing is ever created), a bad *commit* fails only after the clone
    # step itself has already populated the temporary directory -- proving the cleanup in
    # `finally` actually removes it, not just that nothing was ever there to begin with.
    repo, _, _ = upstream_repo
    bad_commit = "f" * 40
    card = parse_card(_gated_card_yaml("pinned-test", repo, bad_commit))

    with pytest.raises(ContractError, match="does not exist"):
        ensure_clone(card)

    dest = clone_cache_dir(card.name, bad_commit)
    assert not dest.exists()
    leftover_tmp = [p for p in dest.parent.iterdir() if p.name.startswith(f".{dest.name}.tmp-")]
    assert leftover_tmp == []


def test_ensure_clone_removes_a_stale_tmp_directory_from_a_previous_crash(isolated, upstream_repo):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = clone_cache_dir(card.name, commit)
    stale_tmp = dest.parent / f".{dest.name}.tmp-{os.getpid()}"
    stale_tmp.mkdir(parents=True)
    (stale_tmp / "leftover-from-a-crash").write_text("x")

    result = ensure_clone(card)

    assert result == dest
    assert (dest / "src" / "model.py").is_file()
    assert not stale_tmp.exists()


def test_ensure_clone_reuses_a_concurrently_created_clone(isolated, upstream_repo, monkeypatch):
    """Simulates losing a race with another process: `dest` is pre-created (with a valid clone)
    right in the middle of this call's own attempt, so its `tmp.replace(dest)` fails because
    `dest` now exists. The temp clone must be discarded and the winner's `dest` reused instead."""
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = clone_cache_dir(card.name, commit)
    real_clone_at_commit = strategies_module.clone_at_commit

    def racing_clone(repo_arg: str, commit_arg: str, tmp_dest: Path) -> Path:
        if not dest.is_dir():
            real_clone_at_commit(repo_arg, commit_arg, dest)  # the "other process" wins first
        return real_clone_at_commit(repo_arg, commit_arg, tmp_dest)

    monkeypatch.setattr(strategies_module, "clone_at_commit", racing_clone)

    result = ensure_clone(card)

    assert result == dest
    assert (dest / "src" / "model.py").is_file()
    leftover_tmp = [p for p in dest.parent.iterdir() if p.name.startswith(f".{dest.name}.tmp-")]
    assert leftover_tmp == []


def test_ensure_clone_refuses_a_pre_created_dest_that_fails_its_own_checks(
    isolated, upstream_repo, monkeypatch
):
    """The same race, but the process that won it left something bad behind: the loser must still
    refuse, not silently accept whatever is sitting at `dest`."""
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test", repo, commit))
    dest = clone_cache_dir(card.name, commit)
    real_clone_at_commit = strategies_module.clone_at_commit

    def racing_clone_with_a_dirty_winner(repo_arg: str, commit_arg: str, tmp_dest: Path) -> Path:
        if not dest.is_dir():
            real_clone_at_commit(repo_arg, commit_arg, dest)
            (dest / "extra.txt").write_text("uncommitted")
        return real_clone_at_commit(repo_arg, commit_arg, tmp_dest)

    monkeypatch.setattr(strategies_module, "clone_at_commit", racing_clone_with_a_dirty_winner)

    with pytest.raises(ContractError, match="local modifications"):
        ensure_clone(card)


def test_ensure_clone_is_gated_by_licence(isolated, upstream_repo, monkeypatch):
    repo, commit, _ = upstream_repo
    card = parse_card(_gated_card_yaml("pinned-test-gated", repo, commit, requires_ack=True))

    def boom(*args, **kwargs):
        raise AssertionError("clone_at_commit must not run before the licence is accepted")

    # Scoped to just this one patch: a bare `monkeypatch.undo()` here would also revert the
    # autouse `isolated` fixture's own DFWB_CACHE_ROOT/DFWB_STATE_DIR patches (they share the
    # same function-scoped MonkeyPatch), which would then send the acceptance and the real clone
    # below to the machine's real cache and state directories instead of tmp_path.
    with monkeypatch.context() as gate_check:
        gate_check.setattr(strategies_module, "clone_at_commit", boom)
        with pytest.raises(InstallationError) as info:
            ensure_clone(card)
        assert info.value.exit_code == 5
        assert not clone_cache_dir(card.name, commit).exists()

    licenses.accept(card.name, license=card.license.code)
    dest = ensure_clone(card)
    assert (dest / "src" / "model.py").is_file()


def test_ensure_clone_requires_an_upstream_block(isolated):
    card = parse_card(
        """
name: no-upstream
display_name: No Upstream
contract_version: [1, 0]
license: {code: MIT}
code_strategy: pinned-clone
input: {}
"""
    )
    with pytest.raises(ConfigError, match="upstream"):
        ensure_clone(card)


# ---------------------------------------------------------------------------- import_pinned_entry


def _card_named(name: str) -> AdapterCard:
    """A minimal, valid card with the given ``name`` -- ``import_pinned_entry`` derives the
    private module name from it, so these tests never pass a bare string of their own."""
    return parse_card(
        f"""
name: {name}
display_name: Test
contract_version: [1, 0]
license: {{code: MIT}}
code_strategy: pip
input: {{}}
"""
    )


def _forget(module_name: str, *extra: str) -> None:
    """Pop ``module_name`` and everything under it, plus any ``extra`` bare names (and their own
    submodules), from ``sys.modules`` -- test teardown for the private-package tests below, which
    otherwise leak into later tests since ``sys.modules`` is global, process-wide state."""
    prefixes = [f"{module_name}."] + [f"{name}." for name in extra]
    for key in list(sys.modules):
        if key == module_name or key in extra or any(key.startswith(p) for p in prefixes):
            sys.modules.pop(key, None)


def test_private_module_name_is_sanitised():
    assert private_module_name("gend") == "dfwb_zoo_ext_gend"
    assert private_module_name("My/Weird Name") == "dfwb_zoo_ext_my_weird_name"


def test_leak_detection_requires_a_real_package_or_module_at_the_clone_root(tmp_path):
    (tmp_path / "datasets").mkdir()  # a plain data folder: no __init__.py, not a package
    (tmp_path / "realpkg").mkdir()
    (tmp_path / "realpkg" / "__init__.py").write_text("")
    (tmp_path / "realmodule.py").write_text("")

    is_importable = strategies_module._is_importable_at_clone_root
    assert is_importable(tmp_path, "datasets") is False
    assert is_importable(tmp_path, "realpkg") is True
    assert is_importable(tmp_path, "realmodule") is True
    assert is_importable(tmp_path, "nonexistent") is False


def test_a_clone_root_with_its_own_init_file_is_loaded(tmp_path):
    (tmp_path / "__init__.py").write_text("PACKAGE_MARKER = 'root init ran'\n")
    (tmp_path / "leaf.py").write_text("VALUE = 1\n")
    card = _card_named("root-init-test")
    module_name = private_module_name(card.name)

    try:
        module = import_pinned_entry(tmp_path, card, "leaf")
        assert module.VALUE == 1
        assert sys.modules[module_name].PACKAGE_MARKER == "root init ran"
    finally:
        _forget(module_name)


def test_a_raising_root_init_file_is_reported(tmp_path):
    (tmp_path / "__init__.py").write_text("raise RuntimeError('root init kaboom')\n")
    card = _card_named("bad-root-init-test")
    module_name = private_module_name(card.name)

    with pytest.raises(ContractError, match="root init kaboom"):
        import_pinned_entry(tmp_path, card, "leaf")
    assert module_name not in sys.modules


def test_registering_the_same_private_package_twice_reuses_it(tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("reuse-test")
    module_name = private_module_name(card.name)

    try:
        first = import_pinned_entry(clone_root, card, "src.model")
        second = import_pinned_entry(clone_root, card, "src.model")
        assert first is second
        assert second.make_detector() == 142
    finally:
        _forget(module_name)


def test_reregisters_when_the_same_card_points_at_a_different_clone_root(tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    clone_a = clone_at_commit(str(repo), commit, tmp_path / "clone-a")
    card = _card_named("reregister-test")
    module_name = private_module_name(card.name)

    clone_b = tmp_path / "clone-b"
    clone_b.mkdir()
    (clone_b / "__init__.py").write_text("")
    (clone_b / "leaf.py").write_text("VALUE = 999\n")

    try:
        first = import_pinned_entry(clone_a, card, "src.model")
        assert first.make_detector() == 142

        second = import_pinned_entry(clone_b, card, "leaf")
        assert second.VALUE == 999
        assert list(sys.modules[module_name].__path__) == [str(clone_b)]
        # The stale registration (and anything cached under it from clone_a) is gone, not just
        # shadowed -- a later re-import of the old entry would not find it lying around.
        assert f"{module_name}.src" not in sys.modules
    finally:
        _forget(module_name)


def test_a_relative_import_inside_the_clone_resolves_correctly(tmp_path, upstream_repo):
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("relative-test")
    module_name = private_module_name(card.name)

    assert "src" not in sys.modules
    try:
        module = import_pinned_entry(clone_root, card, "src.model")
        assert module.make_detector() == 142  # 42 + 100: the relative import really resolved
        assert "src" not in sys.modules
        assert f"{module_name}.src" in sys.modules
    finally:
        _forget(module_name)


def test_an_absolute_self_import_that_collides_is_refused(tmp_path, upstream_repo, monkeypatch):
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("absolute-test")
    module_name = private_module_name(card.name)

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "src.py").write_text("MARKER = 'decoy'\n")
    monkeypatch.syspath_prepend(str(decoy_dir))
    path_before = list(sys.path)

    assert "src" not in sys.modules
    try:
        with pytest.raises(ContractError, match="pulled in") as info:
            import_pinned_entry(clone_root, card, "absolute_entry")
        assert "vendor a small shim" in info.value.hint
        assert "src" not in sys.modules  # popped, not left holding the wrong module
        assert f"{module_name}.absolute_entry" not in sys.modules
        assert sys.path == path_before  # never touched
    finally:
        _forget(module_name, "src")


def test_a_refused_import_does_not_stick_around_for_a_retry(tmp_path, upstream_repo, monkeypatch):
    """A second call, with the same colliding decoy still in place, must detect the leak again --
    not silently return whatever the first, refused attempt left cached in ``sys.modules``."""
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("retry-test")
    module_name = private_module_name(card.name)

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "src.py").write_text("MARKER = 'decoy'\n")
    monkeypatch.syspath_prepend(str(decoy_dir))

    try:
        with pytest.raises(ContractError, match="pulled in"):
            import_pinned_entry(clone_root, card, "absolute_entry")
        assert f"{module_name}.absolute_entry" not in sys.modules

        with pytest.raises(ContractError, match="pulled in"):
            import_pinned_entry(clone_root, card, "absolute_entry")
        assert f"{module_name}.absolute_entry" not in sys.modules
    finally:
        _forget(module_name, "src")


def test_a_leak_found_on_the_exception_path_gets_the_vendoring_hint(
    tmp_path, upstream_repo, monkeypatch
):
    """``from src.helper import ...`` against a plain-module decoy ``src.py`` (not a package)
    raises -- but only after ``src`` itself has already been bound to the decoy, which must still
    be caught and reported with the vendoring hint, not treated as just "a bug in the adapter"."""
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("exception-leak-test")
    module_name = private_module_name(card.name)

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "src.py").write_text("MARKER = 'decoy'\n")
    monkeypatch.syspath_prepend(str(decoy_dir))

    assert "src" not in sys.modules
    try:
        with pytest.raises(ContractError, match="pulled in") as info:
            import_pinned_entry(clone_root, card, "absolute_from_helper_entry")
        assert "vendor a small shim" in info.value.hint
        assert "src" not in sys.modules
        assert "src.helper" not in sys.modules
    finally:
        _forget(module_name, "src")


def test_a_leaked_packages_own_submodules_are_also_purged(tmp_path, upstream_repo, monkeypatch):
    """A decoy that is itself a *package* (not a plain module) succeeds outright -- ``src`` and
    ``src.helper`` both resolve against it -- and both must be purged, not just the bare name."""
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("submodule-leak-test")
    module_name = private_module_name(card.name)

    decoy_dir = tmp_path / "decoy_pkg"
    (decoy_dir / "src").mkdir(parents=True)
    (decoy_dir / "src" / "__init__.py").write_text("")
    (decoy_dir / "src" / "helper.py").write_text("HELPER_VALUE = -1\n")
    monkeypatch.syspath_prepend(str(decoy_dir))

    assert "src" not in sys.modules
    assert "src.helper" not in sys.modules
    try:
        with pytest.raises(ContractError, match="pulled in"):
            import_pinned_entry(clone_root, card, "absolute_from_helper_entry")
        assert "src" not in sys.modules
        assert "src.helper" not in sys.modules
    finally:
        _forget(module_name, "src")


def test_a_name_already_imported_before_the_call_binds_silently_known_limitation(
    tmp_path, upstream_repo, monkeypatch
):
    """Documented limitation (see the module docstring and ``import_pinned_entry``'s own): the
    before/after ``sys.modules`` diff can only see a name that is newly imported *during* this
    call. A colliding name already present beforehand -- imported by anything else, earlier in the
    same process -- is invisible to it, and an absolute self-import inside the clone silently
    binds to that already-cached module instead of being refused."""
    repo, commit, _ = upstream_repo
    clone_root = clone_at_commit(str(repo), commit, tmp_path / "clone")
    card = _card_named("preexisting-src-test")
    module_name = private_module_name(card.name)

    decoy_dir = tmp_path / "decoy"
    decoy_dir.mkdir()
    (decoy_dir / "src.py").write_text("MARKER = 'decoy'\n")
    monkeypatch.syspath_prepend(str(decoy_dir))
    import src as _pre_existing_src  # noqa: F401 -- imported before the call, on purpose

    try:
        module = import_pinned_entry(clone_root, card, "absolute_entry")
        assert module.MARKER == "decoy"  # the known gap: silently bound to the wrong module
    finally:
        _forget(module_name, "src")


def test_an_unloadable_entry_is_reported(tmp_path, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "spec_from_loader", lambda *a, **k: None)
    card = _card_named("unloadable-test")
    with pytest.raises(ContractError, match="cannot be loaded"):
        import_pinned_entry(tmp_path, card, "whatever")


def test_a_raising_entry_is_reported_and_not_left_in_sys_modules(tmp_path):
    (tmp_path / "bad.py").write_text("raise RuntimeError('kaboom')\n")
    card = _card_named("bad-test")
    module_name = private_module_name(card.name)

    with pytest.raises(ContractError, match="kaboom"):
        import_pinned_entry(tmp_path, card, "bad")
    assert f"{module_name}.bad" not in sys.modules
