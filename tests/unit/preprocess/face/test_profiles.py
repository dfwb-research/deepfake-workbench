import re

import pytest
import yaml

from dfwb.core.errors import ContractError, UnknownKeyError
from dfwb.core.records import ProcessingProfile
from dfwb.preprocess.face.profiles import builtin_profiles, load_profile


def test_builtin_profiles_lists_every_shipped_profile():
    names = builtin_profiles()
    assert names == sorted(names)
    assert {
        "face-256-1.3x-64f",
        "face-256-1.3x-32f",
        "face-256-1.3x-64fc",
        "face-256-1.3x-32f-mp",
        "toy-64-center-8f",
    } <= set(names)


@pytest.mark.parametrize(
    "name",
    [
        "face-256-1.3x-64f",
        "face-256-1.3x-32f",
        "face-256-1.3x-64fc",
        "face-256-1.3x-32f-mp",
        "toy-64-center-8f",
    ],
)
def test_every_builtin_profile_loads_and_has_a_stable_id(name):
    profile = load_profile(name)
    assert isinstance(profile, ProcessingProfile)
    assert profile.id == name
    first = profile.profile_id()
    second = load_profile(name).profile_id()
    assert first == second


def test_parity_profile_has_exactly_the_expected_values():
    profile = load_profile("face-256-1.3x-64f")
    assert profile.backend.name == "insightface"
    assert profile.backend.model_extra == {"model": "buffalo_l", "det_size": 256, "min_score": 0.5}
    assert profile.track.iou == 0.3
    assert profile.track.strategy == "largest-then-iou"
    assert profile.track.ema is None
    assert profile.crop.scale == 1.3
    assert profile.crop.size == 256
    assert profile.crop.square is True
    assert profile.sampling.mode == "uniform"
    assert profile.sampling.frames == 64
    assert profile.decode.library == "opencv"
    assert profile.extras.mesh is False
    assert profile.extras.masks is False


def test_profile_hash_does_not_depend_on_yaml_key_order():
    profile = load_profile("face-256-1.3x-64f")
    reordered = dict(reversed(list(profile.model_dump(mode="json").items())))
    dumped = yaml.safe_dump(reordered, sort_keys=False)
    again = ProcessingProfile.model_validate(yaml.safe_load(dumped))
    assert again.sha256() == profile.sha256()


def test_load_profile_reads_a_path(tmp_path):
    text = yaml.safe_dump(load_profile("toy-64-center-8f").model_dump(mode="json"))
    path = tmp_path / "custom.yaml"
    path.write_text(text)
    profile = load_profile(str(path))
    assert profile.id == "toy-64-center-8f"


def test_load_profile_rejects_invalid_yaml_as_contract_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("id: [unterminated\n")
    with pytest.raises(ContractError):
        load_profile(str(path))


def test_load_profile_rejects_a_schema_violation_as_contract_error(tmp_path):
    path = tmp_path / "bad-shape.yaml"
    path.write_text("id: not-a-profile\n")
    with pytest.raises(ContractError, match="required key is missing"):
        load_profile(str(path))


def test_unknown_profile_name_raises_with_did_you_mean():
    with pytest.raises(UnknownKeyError, match=re.escape("face-256-1.3x-64f")):
        load_profile("face-256-1.3x-64")
