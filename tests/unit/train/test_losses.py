"""Losses (``losses`` registry): ``forward(out, batch) -> LossOutput(total, parts)``.

Every expected value below is computed independently of ``dfwb.train.losses`` (plain ``math``,
never by calling the code under test), so a hand-computation mismatch is a real bug, not a
tautology.
"""

from __future__ import annotations

import math

import pytest
import torch

from dfwb.core.detector import ClipBatch, DetectorOutput
from dfwb.core.errors import ConfigError, ContractError
from dfwb.core.plugins import api, get_registry
from dfwb.train.losses import BCELoss, CELoss, FocalLoss, LabelSmoothingBCELoss, LossOutput


def _batch(labels: torch.Tensor | None) -> ClipBatch:
    b = labels.shape[0] if labels is not None else 1
    return ClipBatch(
        clips=torch.zeros(b, 1, 3, 4, 4),
        keys=[f"k{i}" for i in range(b)],
        dataset_ids=["toy"] * b,
        compressions=[None] * b,
        clip_index=torch.zeros(b, dtype=torch.long),
        frame_indices=torch.zeros(b, 1, dtype=torch.long),
        labels=labels,
    )


def _binary_out(logit: torch.Tensor) -> DetectorOutput:
    return DetectorOutput(score=torch.sigmoid(logit), logit=logit)


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _stable_bce_term(z: float, y: float) -> float:
    # The numerically stable form F.binary_cross_entropy_with_logits itself uses.
    return max(z, 0.0) - z * y + math.log(1 + math.exp(-abs(z)))


# --------------------------------------------------------------------------- LossOutput


def test_loss_output_is_a_frozen_dataclass():
    out = LossOutput(total=torch.tensor(1.0), parts={"x": torch.tensor(1.0)})
    with pytest.raises(AttributeError):
        out.total = torch.tensor(2.0)  # type: ignore[misc]


# --------------------------------------------------------------------------- bce


def test_bce_matches_hand_computation():
    logit = torch.tensor([2.0, -1.0, 0.0])
    labels = torch.tensor([1.0, 0.0, 1.0])
    out = BCELoss()(_binary_out(logit), _batch(labels))
    expected = (
        sum(_stable_bce_term(z, y) for z, y in zip(logit.tolist(), labels.tolist(), strict=True))
        / 3
    )
    assert isinstance(out, LossOutput)
    assert out.total.item() == pytest.approx(expected, rel=1e-5)
    assert out.parts.keys() == {"bce"}
    assert out.parts["bce"] is out.total


def test_bce_pos_weight_scales_only_the_positive_term():
    logit = torch.tensor([0.0, 0.0])
    labels = torch.tensor([1.0, 0.0])
    weighted = BCELoss(pos_weight=3.0)(_binary_out(logit), _batch(labels)).total
    term = -math.log(_sigmoid(0.0))  # sigmoid(0) = 0.5 for both the positive and negative term
    expected = (3.0 * term + term) / 2
    assert weighted.item() == pytest.approx(expected, rel=1e-5)


def test_bce_missing_labels_raises_contract_error():
    with pytest.raises(ContractError, match="labels"):
        BCELoss()(_binary_out(torch.tensor([0.0])), _batch(None))


def test_bce_missing_logit_raises_contract_error():
    out = DetectorOutput(score=torch.tensor([0.5]), logit=None)
    with pytest.raises(ContractError, match="logit"):
        BCELoss()(out, _batch(torch.tensor([1.0])))


# --------------------------------------------------------------------------- ce


def test_ce_matches_hand_computation_with_label_smoothing():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    labels = torch.tensor([0.0, 1.0])
    eps = 0.1
    out = CELoss(label_smoothing=eps)(
        DetectorOutput(score=torch.zeros(2), logit=logits), _batch(labels)
    )

    def smoothed_ce(row: list[float], target: int, k: int) -> float:
        m = max(row)
        log_z = m + math.log(sum(math.exp(x - m) for x in row))
        log_probs = [x - log_z for x in row]
        weight_true = 1 - eps + eps / k
        weight_other = eps / k
        other = sum(log_probs[i] for i in range(k) if i != target)
        return -(weight_true * log_probs[target] + weight_other * other)

    expected = (smoothed_ce(logits[0].tolist(), 0, 3) + smoothed_ce(logits[1].tolist(), 1, 3)) / 2
    assert out.total.item() == pytest.approx(expected, rel=1e-4)
    assert out.parts.keys() == {"ce"}


def test_ce_binary_logit_is_a_config_error_naming_bce():
    logit = torch.tensor([0.0, 1.0])
    with pytest.raises(ConfigError, match="bce"):
        CELoss()(_binary_out(logit), _batch(torch.tensor([0.0, 1.0])))


def test_ce_missing_labels_raises_contract_error():
    logits = torch.zeros(2, 3)
    with pytest.raises(ContractError, match="labels"):
        CELoss()(DetectorOutput(score=torch.zeros(2), logit=logits), _batch(None))


# --------------------------------------------------------------------------- focal


def test_focal_gamma_zero_alpha_none_equals_bce():
    logit = torch.tensor([2.0, -1.0, 0.3])
    labels = torch.tensor([1.0, 0.0, 1.0])
    bce = BCELoss()(_binary_out(logit), _batch(labels)).total
    focal = FocalLoss(alpha=None, gamma=0.0)(_binary_out(logit), _batch(labels)).total
    assert focal.item() == pytest.approx(bce.item(), rel=1e-6)


def test_focal_matches_hand_computation():
    logit = torch.tensor([1.0, -0.5])
    labels = torch.tensor([1.0, 0.0])
    alpha, gamma = 0.25, 2.0
    out = FocalLoss(alpha=alpha, gamma=gamma)(_binary_out(logit), _batch(labels))

    def term(z: float, y: float) -> float:
        p = _sigmoid(z)
        p_t = p * y + (1 - p) * (1 - y)
        ce = _stable_bce_term(z, y)
        alpha_t = alpha * y + (1 - alpha) * (1 - y)
        return alpha_t * ((1 - p_t) ** gamma) * ce

    expected = sum(term(z, y) for z, y in zip(logit.tolist(), labels.tolist(), strict=True)) / 2
    assert out.total.item() == pytest.approx(expected, rel=1e-5)


def test_focal_missing_labels_raises_contract_error():
    with pytest.raises(ContractError, match="labels"):
        FocalLoss()(_binary_out(torch.tensor([0.0])), _batch(None))


# --------------------------------------------------------------------------- label-smoothing-bce


def test_label_smoothing_bce_matches_hand_computation():
    logit = torch.tensor([0.5, -0.5])
    labels = torch.tensor([1.0, 0.0])
    eps = 0.2
    out = LabelSmoothingBCELoss(eps=eps)(_binary_out(logit), _batch(labels))
    smoothed = [y * (1 - eps) + eps / 2 for y in labels.tolist()]
    expected = (
        sum(_stable_bce_term(z, y) for z, y in zip(logit.tolist(), smoothed, strict=True)) / 2
    )
    assert out.total.item() == pytest.approx(expected, rel=1e-5)
    assert out.parts.keys() == {"label-smoothing-bce"}


def test_label_smoothing_bce_default_eps_is_0_1():
    assert LabelSmoothingBCELoss().eps == 0.1


def test_label_smoothing_bce_missing_labels_raises_contract_error():
    with pytest.raises(ContractError, match="labels"):
        LabelSmoothingBCELoss()(_binary_out(torch.tensor([0.0])), _batch(None))


# --------------------------------------------------------------------------- registry


@pytest.mark.parametrize(
    ("key", "cls"),
    [
        ("bce", BCELoss),
        ("ce", CELoss),
        ("focal", FocalLoss),
        ("label-smoothing-bce", LabelSmoothingBCELoss),
    ],
)
def test_registered_as_a_builtin_loss(key, cls):
    entry = get_registry("losses").entry(key)
    assert entry.provider == "dfwb"
    assert entry.requires == ("torch",)
    loss = api.losses.build(key)
    assert isinstance(loss, cls)


def test_unknown_loss_param_is_a_config_error_through_the_registry():
    with pytest.raises(ConfigError, match="unknown parameter"):
        api.losses.build("bce", pos_wieght=1.0)
