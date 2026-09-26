"""Code strategies: the ``vendored`` layout checker, and ``pinned-clone`` against a local git
fixture repository (never the real network) whose top-level ``src/`` package must never reach
``sys.modules`` -- only the adapter's own private module name does."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from dfwb.core.errors import ContractError, InstallationError
from dfwb.zoo.strategies import check_vendored_layout, clone_at_commit, load_pinned_module

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


# ------------------------------------------------------------------------------------ pinned-clone


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def upstream_repo(tmp_path: Path) -> tuple[Path, str]:
    """A local git repository with a top-level ``src/`` package, so cloning and importing it is
    exactly the case that used to force vendoring code just to dodge a ``sys.modules`` collision."""
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.org"], repo)
    _git(["config", "user.name", "Test"], repo)
    (repo / "src").mkdir()
    (repo / "src" / "__init__.py").write_text("")
    (repo / "src" / "model.py").write_text(
        "VALUE = 42\n\n\ndef make_detector():\n    return VALUE\n"
    )
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "initial"], repo)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, commit


def test_clone_at_commit_checks_out_exactly_that_commit(tmp_path, upstream_repo):
    repo, commit = upstream_repo
    dest = tmp_path / "clone"

    result = clone_at_commit(str(repo), commit, dest)

    assert result == dest
    assert (dest / "src" / "model.py").is_file()
    checked_out = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert checked_out == commit


def test_clone_at_commit_rejects_a_nonexistent_commit(tmp_path, upstream_repo):
    repo, _ = upstream_repo
    with pytest.raises(ContractError, match="does not exist"):
        clone_at_commit(str(repo), "f" * 40, tmp_path / "clone")


def test_clone_at_commit_rejects_a_nonexistent_repo(tmp_path):
    with pytest.raises(InstallationError):
        clone_at_commit(str(tmp_path / "no-such-repo"), "f" * 40, tmp_path / "clone")


def test_clone_at_commit_reports_a_missing_git_binary(tmp_path, upstream_repo, monkeypatch):
    repo, commit = upstream_repo
    monkeypatch.setenv("PATH", "")  # git cannot be found on an empty PATH
    with pytest.raises(InstallationError, match="git is not installed"):
        clone_at_commit(str(repo), commit, tmp_path / "clone")


def test_a_cloned_top_level_src_package_never_reaches_sys_modules(tmp_path, upstream_repo):
    repo, commit = upstream_repo
    dest = clone_at_commit(str(repo), commit, tmp_path / "clone")
    private_name = "dfwb_zoo_ext_gend_test"

    assert "src" not in sys.modules
    try:
        module = load_pinned_module(dest / "src" / "model.py", private_name)
        assert "src" not in sys.modules
        assert private_name in sys.modules
        assert module.make_detector() == 42
    finally:
        sys.modules.pop(private_name, None)


def test_an_unloadable_entry_file_is_reported(tmp_path, monkeypatch):
    import importlib.util

    from dfwb.zoo import strategies

    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)
    with pytest.raises(ContractError, match="cannot be loaded"):
        strategies.load_pinned_module(tmp_path / "whatever.py", "dfwb_zoo_ext_unloadable")


def test_a_raising_entry_file_is_reported_and_not_left_in_sys_modules(tmp_path):
    entry = tmp_path / "bad.py"
    entry.write_text("raise RuntimeError('kaboom')\n")
    private_name = "dfwb_zoo_ext_bad_test"

    with pytest.raises(ContractError, match="kaboom"):
        load_pinned_module(entry, private_name)
    assert private_name not in sys.modules
