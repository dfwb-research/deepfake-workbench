import dataclasses

import pytest

from dfwb.core.detector import (
    DETECTOR_CONTRACT_VERSION,
    ClipBatch,
    Detector,
    DetectorMeta,
    DetectorOutput,
    InputSpec,
)
from dfwb.core.errors import ContractError


def test_input_spec_defaults_match_contract():
    spec = InputSpec()
    assert (spec.modality, spec.crop, spec.crop_scale, spec.size, spec.frames) == (
        "frames",
        "face",
        1.3,
        (224, 224),
        1,
    )
    assert (spec.sampling, spec.color, spec.value_range, spec.mean, spec.std) == (
        "any",
        "rgb",
        (0.0, 1.0),
        None,
        None,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"crop_scale": 0}, "crop_scale"),
        ({"size": (224, 0)}, "size"),
        ({"frames": 0}, "frames"),
        ({"value_range": (1.0, 0.0)}, "value_range"),
        ({"mean": (0.5,)}, "given together"),
        ({"mean": (0.5, 0.5), "std": (0.2,)}, "same length"),
        ({"mean": (0.5,), "std": (0.0,)}, "positive"),
    ],
)
def test_input_spec_rejects_impossible_values(kwargs, message):
    with pytest.raises(ContractError, match=message):
        InputSpec(**kwargs)


def test_meta_is_frozen_and_versioned():
    meta = DetectorMeta(
        "tiny", "0.1", DETECTOR_CONTRACT_VERSION, InputSpec(), "MIT", None, None, "run:abc"
    )
    assert meta.training_data == ()
    assert DETECTOR_CONTRACT_VERSION == (1, 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.name = "other"  # type: ignore[misc]


def test_detector_protocol_is_structural():
    class Chance:
        meta = DetectorMeta("chance", "0.1", (1, 0), InputSpec(), "MIT", None, None, None)

        def to(self, device):
            return self

        def predict(self, batch):
            return DetectorOutput(score=[0.5] * len(batch.keys))

    batch = ClipBatch(
        clips=None,
        keys=["a", "b"],
        dataset_ids=["d", "d"],
        compressions=[None, None],
        clip_index=None,
        frame_indices=None,
    )
    assert isinstance(Chance(), Detector)
    assert Chance().predict(batch).score == [0.5, 0.5]
    assert batch.extras == {}
    assert not isinstance(object(), Detector)
