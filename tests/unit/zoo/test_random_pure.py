"""``random``'s pure scoring function: deterministic in ``(seed, dataset, key, compression)``
alone, independent of call order. Needs no batch, no tensors and no torch at all, so
order-independence stays covered in the torch-free job even when the rest of the adapter (whose
``predict()`` builds a tensor) cannot run there."""

from __future__ import annotations

from dfwb.zoo.adapters.random import _uniform


def test_the_same_inputs_always_give_the_same_value():
    assert _uniform(0, "toy", "real/00", None) == _uniform(0, "toy", "real/00", None)


def test_different_seeds_give_different_values():
    assert _uniform(0, "toy", "real/00", None) != _uniform(1, "toy", "real/00", None)


def test_different_keys_give_different_values():
    assert _uniform(0, "toy", "real/00", None) != _uniform(0, "toy", "real/01", None)


def test_different_datasets_give_different_values():
    assert _uniform(0, "toy", "real/00", None) != _uniform(0, "other", "real/00", None)


def test_compression_is_part_of_the_identity():
    assert _uniform(0, "toy", "real/00", None) != _uniform(0, "toy", "real/00", "c23")


def test_values_are_in_the_unit_interval():
    for i in range(20):
        value = _uniform(i, "toy", f"real/{i:02d}", None)
        assert 0.0 <= value < 1.0


def test_order_of_calls_does_not_affect_the_result():
    # Each call is a pure function of its own arguments alone -- interleaving calls for two keys
    # in either order must give each key exactly the value it would get on its own.
    forward = [_uniform(0, "toy", key, None) for key in ("a", "b")]
    backward = [_uniform(0, "toy", key, None) for key in ("b", "a")]
    assert forward == list(reversed(backward))
