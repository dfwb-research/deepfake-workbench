"""The ``Face`` record a detection backend returns for each face it finds."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dfwb.preprocess.face.types import Face


def test_face_defaults_leave_the_optional_signals_empty():
    face = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9)
    assert face.landmarks5 is None
    assert face.embedding is None
    assert face.yaw is None


def test_face_is_frozen():
    face = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        face.score = 0.1  # type: ignore[misc]


def test_comparing_faces_with_embeddings_never_asks_numpy_for_a_truth_value():
    # Comparing two numpy arrays with == gives an array, and bool() of that raises; the embedding
    # is left out of equality so comparing (or hashing) faces is always safe.
    first = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, embedding=np.ones(4, dtype=np.float32))
    second = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, embedding=np.ones(4, dtype=np.float32))
    other = Face(bbox=(1.0, 2.0, 3.0, 5.0), score=0.9, embedding=np.ones(4, dtype=np.float32))
    assert first == second
    assert first != other
    assert first in [other, second]
    assert hash(first) == hash(second)


def test_the_embedding_does_not_take_part_in_equality():
    first = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, embedding=np.zeros(4, dtype=np.float32))
    second = Face(bbox=(1.0, 2.0, 3.0, 4.0), score=0.9, embedding=None)
    assert first == second
