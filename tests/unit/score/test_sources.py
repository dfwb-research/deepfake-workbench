"""``dfwb.score.sources``: resolving a ``<scheme>:<rest>`` detector URI through the
``detector_sources`` registry, and the ``py:`` source specifically."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

pytest.importorskip("torch")

from tests.unit.score._toy import PROTOCOL, toy_profile, write_toy_store

from dfwb.core.errors import ConfigError, ContractError, UnknownKeyError
from dfwb.core.records.scores import read_scores
from dfwb.score.harness import score
from dfwb.score.sources import resolve_detector


def test_resolves_a_registered_scheme(scoretoy_pack):
    detector = resolve_detector("fake:")
    assert detector.meta.name == "fake-detector"


def test_passes_the_uri_tail_to_the_scheme_loader(scoretoy_pack):
    detector = resolve_detector("fake:scale=1.6")
    assert detector.meta.input.crop_scale == 1.6


def test_unknown_scheme_lists_the_known_schemes(scoretoy_pack):
    with pytest.raises(UnknownKeyError) as info:
        resolve_detector("bogus:whatever")
    assert "bogus" in info.value.message
    assert "run" in info.value.hint
    assert "fake" in info.value.hint


def test_a_uri_with_no_scheme_is_rejected():
    with pytest.raises(UnknownKeyError):
        resolve_detector("no-colon-here")


# ------------------------------------------------------------------------------------------ py:

# A factory that builds a full C4 detector by hand (no torch weights, just the mean of the
# already-adapted pixel values, in [0, 1]) -- enough to score a toy store end to end.
_VALID_FACTORY = """
from __future__ import annotations

from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, DetectorOutput, InputSpec


class _Detector:
    def __init__(self) -> None:
        self.meta = DetectorMeta(
            name="py-factory-detector",
            version="0",
            contract_version=DETECTOR_CONTRACT_VERSION,
            input=InputSpec(crop="face", crop_scale=1.3, size=(32, 32), frames=1),
            license="MIT",
            weights_license=None,
            citation=None,
            source=None,
        )

    def to(self, device):
        return self

    def predict(self, batch):
        mean = batch.clips.mean(dim=tuple(range(1, batch.clips.ndim))).clamp(0.0, 1.0)
        return DetectorOutput(score=mean)


def make_detector():
    return _Detector()
"""

# Returns a plain object with none of meta, predict or to.
_NOTHING_FACTORY = """
def make_nothing():
    return object()
"""

# Returns an object with meta but neither predict nor to.
_META_ONLY_FACTORY = """
from dfwb.core.detector import DETECTOR_CONTRACT_VERSION, DetectorMeta, InputSpec


class _MetaOnly:
    def __init__(self) -> None:
        self.meta = DetectorMeta(
            name="half-detector",
            version="0",
            contract_version=DETECTOR_CONTRACT_VERSION,
            input=InputSpec(),
            license="MIT",
            weights_license=None,
            citation=None,
            source=None,
        )


def make_half():
    return _MetaOnly()
"""

# A module with no callable factory at all.
_NO_FACTORY = """
answer = 42
"""

# Overwrites its own __file__ at import time, the way a namespace package or a compiled
# extension has none to begin with -- distinct from a file that existed and then went missing.
# (Appended, not prepended: "from __future__ import ..." must stay the first statement.)
_NO_FILE_FACTORY = _VALID_FACTORY + "\n__file__ = None\n"


def _write_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> str:
    """Write ``body`` as a freshly, uniquely named top-level module under ``tmp_path`` (put on
    ``sys.path``) and return its module name. A fresh name every call means two tests -- or two
    calls within the same test -- never collide in Python's own module cache
    (:data:`sys.modules`), the way two source files written to the same *path* would."""
    monkeypatch.syspath_prepend(str(tmp_path))
    name = f"py_factory_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text(body, encoding="utf-8")
    return name


def _score_via_py(tmp_path: Path, module_name: str, factory: str, **kwargs: object):
    kwargs.setdefault("protocol", PROTOCOL)
    kwargs.setdefault("split", "test")
    kwargs.setdefault("out", tmp_path / "out")
    return score(f"py:{module_name}:{factory}", **kwargs)


def test_py_source_scores_a_toy_store(score_roots, tmp_path, monkeypatch):
    write_toy_store(score_roots, toy_profile("toy-face"))
    module_name = _write_module(tmp_path, monkeypatch, _VALID_FACTORY)

    result = _score_via_py(tmp_path, module_name, "make_detector")

    scored = read_scores(result.csv_path)
    assert {row.status for row in scored.rows} == {"ok"}
    for row in scored.rows:
        assert 0.0 <= row.score <= 1.0
    # A module-source sha is not a checkpoint sha (ruling: checkpoint_sha256 stays None for py:).
    assert scored.meta.detector.checkpoint_sha256 is None


def test_py_source_rejects_non_detector(tmp_path, monkeypatch):
    module_name = _write_module(tmp_path, monkeypatch, _NOTHING_FACTORY)

    with pytest.raises(ContractError) as info:
        resolve_detector(f"py:{module_name}:make_nothing")
    assert "meta" in info.value.message
    assert "predict" in info.value.message
    assert "to" in info.value.message


def test_py_source_lists_only_the_attributes_actually_missing(tmp_path, monkeypatch):
    module_name = _write_module(tmp_path, monkeypatch, _META_ONLY_FACTORY)

    with pytest.raises(ContractError) as info:
        resolve_detector(f"py:{module_name}:make_half")
    assert "predict" in info.value.message
    assert "to" in info.value.message
    assert "meta" not in info.value.message


def test_py_source_rejects_a_missing_factory(tmp_path, monkeypatch):
    module_name = _write_module(tmp_path, monkeypatch, _NO_FACTORY)

    with pytest.raises(ConfigError) as info:
        resolve_detector(f"py:{module_name}:make_detector")
    assert "make_detector" in info.value.message


def test_py_source_rejects_a_ref_with_no_factory_part(tmp_path, monkeypatch):
    module_name = _write_module(tmp_path, monkeypatch, _VALID_FACTORY)

    with pytest.raises(ConfigError) as info:
        resolve_detector(f"py:{module_name}")
    assert module_name in info.value.message


def test_py_source_reports_a_module_that_cannot_be_imported():
    with pytest.raises(ConfigError) as info:
        resolve_detector("py:no_such_module_at_all_xyz:make_detector")
    assert "no_such_module_at_all_xyz" in info.value.message


def test_a_module_with_no_readable_file_at_all_is_treated_as_unreadable(
    score_roots, tmp_path, monkeypatch
):
    """A module that never had a source file to begin with (a namespace package, a compiled
    extension) is exactly as uncacheable as one whose file has since gone missing."""
    write_toy_store(score_roots, toy_profile("toy-face"))
    module_name = _write_module(tmp_path, monkeypatch, _NO_FILE_FACTORY)

    first = _score_via_py(tmp_path, module_name, "make_detector")
    second = _score_via_py(tmp_path, module_name, "make_detector")

    assert first.cached is False
    assert second.cached is False
    assert second.csv_path != first.csv_path


def test_an_edit_to_the_module_file_changes_the_cache_key(score_roots, tmp_path, monkeypatch):
    write_toy_store(score_roots, toy_profile("toy-face"))
    module_name = _write_module(tmp_path, monkeypatch, _VALID_FACTORY)

    first = _score_via_py(tmp_path, module_name, "make_detector")
    assert first.cached is False
    # A real edit to the file on disk -- the module is already imported (Python's own module
    # cache), but the source sha is re-read straight from disk on every resolve.
    (tmp_path / f"{module_name}.py").write_text(_VALID_FACTORY + "\n# edited\n", encoding="utf-8")

    second = _score_via_py(tmp_path, module_name, "make_detector")

    assert second.csv_path != first.csv_path
    assert second.cached is False


def test_rerunning_an_unedited_py_module_is_still_cached(score_roots, tmp_path, monkeypatch):
    write_toy_store(score_roots, toy_profile("toy-face"))
    module_name = _write_module(tmp_path, monkeypatch, _VALID_FACTORY)

    first = _score_via_py(tmp_path, module_name, "make_detector")
    second = _score_via_py(tmp_path, module_name, "make_detector")

    assert first.cached is False
    assert second.cached is True
    assert second.csv_path == first.csv_path


def test_an_unreadable_module_source_disables_caching(score_roots, tmp_path, monkeypatch):
    write_toy_store(score_roots, toy_profile("toy-face"))
    module_name = _write_module(tmp_path, monkeypatch, _VALID_FACTORY)
    module_path = tmp_path / f"{module_name}.py"

    first = _score_via_py(tmp_path, module_name, "make_detector")
    module_path.unlink()  # still importable (Python's module cache); its source is now unreadable

    second = _score_via_py(tmp_path, module_name, "make_detector")
    third = _score_via_py(tmp_path, module_name, "make_detector")

    assert first.cached is False
    assert second.cached is False
    assert third.cached is False
    assert second.csv_path != third.csv_path  # a fresh, uncacheable identity every single time
