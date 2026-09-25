"""Finding a clip's main subject by clustering face embeddings.

The clustering is plain numpy, so the framework does not need scipy at runtime; these tests hold
it to scipy's own average-linkage clustering (``linkage`` + ``fcluster`` with a distance cut),
which is what the earlier face pipeline called, and hold the whole subject choice to a
reimplementation of that pipeline's steps written directly against scipy.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from dfwb.preprocess.face.identity import _cosine_average_linkage, cluster_subject
from dfwb.preprocess.face.types import Face

# ---------------------------------------------------------------------------
# Reference implementation, straight from scipy
# ---------------------------------------------------------------------------


def _scipy_linkage(x: np.ndarray) -> np.ndarray:
    distances = pdist(x, metric="cosine")
    # a zero vector has no direction, so its cosine distance is NaN; the earlier pipeline
    # replaced every non-finite distance with the largest possible one (2.0)
    if not np.all(np.isfinite(distances)):
        distances = np.nan_to_num(distances, nan=2.0, posinf=2.0, neginf=2.0)
    return linkage(distances, method="average")


def _scipy_partition(x: np.ndarray, threshold: float) -> set[frozenset[int]]:
    labels = fcluster(_scipy_linkage(x), t=threshold, criterion="distance")
    groups: dict[int, set[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), set()).add(index)
    return {frozenset(group) for group in groups.values()}


def _unit_norm(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return vector
    return (vector / norm).astype(np.float32)


def _reference_subject(faces: list[Face], threshold: float = 0.5) -> np.ndarray | None:
    """The earlier pipeline's subject choice, step by step, with scipy doing the clustering."""
    pool = []
    for face in faces:
        if face.embedding is None:
            continue
        x1, y1, x2, y2 = face.bbox
        area = max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))
        yaw = 0.0 if face.yaw is None else abs(face.yaw)
        pool.append((np.asarray(face.embedding, dtype=np.float32), area, face.score, yaw))
    if not pool:
        return None
    if len(pool) == 1:
        return _unit_norm(pool[0][0].astype(np.float32, copy=True))
    embeddings = np.vstack([entry[0] for entry in pool]).astype(np.float64)
    labels = fcluster(_scipy_linkage(embeddings), t=threshold, criterion="distance")
    clusters: dict[int, list[tuple[np.ndarray, float, float, float]]] = {}
    for entry, label in zip(pool, labels, strict=True):
        clusters.setdefault(int(label), []).append(entry)

    def score(cluster: list[tuple[np.ndarray, float, float, float]]) -> float:
        total = 0.0
        for _, area, det_score, yaw in cluster:
            total += float(area) * float(det_score) * max(0.0, 1.0 - float(yaw) / 90.0)
        return total

    chosen = max(clusters.values(), key=score)
    mean = np.mean(np.vstack([entry[0] for entry in chosen]), axis=0).astype(np.float32)
    return _unit_norm(mean)


# ---------------------------------------------------------------------------
# Random embedding sets
# ---------------------------------------------------------------------------


@st.composite
def _embedding_sets(draw: st.DrawFn) -> np.ndarray:
    """Points scattered around a few directions, so partitions are neither all singletons nor
    one big cluster. Now and then a row is zeroed, to exercise the no-direction case, or copied
    over another row, as happens when a short clip samples the same frame twice."""
    count = draw(st.integers(2, 14))
    dims = draw(st.integers(2, 8))
    centres = draw(st.integers(1, 4))
    spread = draw(st.floats(0.05, 1.2))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((centres, dims))
    points = directions[rng.integers(0, centres, count)]
    points = points + spread * rng.standard_normal((count, dims))
    if draw(st.booleans()):
        source, target = draw(st.integers(0, count - 1)), draw(st.integers(0, count - 1))
        points[target] = points[source]
    if draw(st.booleans()) and draw(st.booleans()):
        points[draw(st.integers(0, count - 1))] = 0.0
    return points


@settings(max_examples=50, deadline=None)
@given(_embedding_sets())
def test_numpy_average_linkage_matches_scipy_as_partitions(points):
    # a merge height within a hair of the cut is decided by rounding, not by the algorithm
    assume(np.all(np.abs(_scipy_linkage(points)[:, 2] - 0.5) > 1e-9))

    clusters = _cosine_average_linkage(points, threshold=0.5)

    assert {frozenset(cluster) for cluster in clusters} == _scipy_partition(points, 0.5)
    assert sorted(i for cluster in clusters for i in cluster) == list(range(len(points)))


@settings(max_examples=50, deadline=None)
@given(
    _embedding_sets(),
    st.lists(
        st.tuples(
            st.floats(1, 200),
            st.floats(0.5, 1.0),
            st.none() | st.floats(-120, 120),
        ),
        min_size=14,
        max_size=14,
    ),
)
def test_cluster_subject_matches_the_scipy_reference(points, extras):
    assume(np.all(np.abs(_scipy_linkage(points)[:, 2] - 0.5) > 1e-9))
    faces = [
        Face(
            bbox=(10.0, 20.0, 10.0 + side, 20.0 + side * 1.1),
            score=score,
            embedding=points[i].astype(np.float32),
            yaw=yaw,
        )
        for i, (side, score, yaw) in enumerate(extras[: len(points)])
    ]
    # float32 embeddings can sit at a slightly different distance from the cut than the
    # float64 points the assumption above looked at
    as_used = np.vstack([face.embedding for face in faces]).astype(np.float64)
    assume(np.all(np.abs(_scipy_linkage(as_used)[:, 2] - 0.5) > 1e-9))

    expected = _reference_subject(faces)
    got = cluster_subject(faces)

    assert expected is not None
    assert got is not None
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, expected)


def test_clusters_come_back_ordered_by_their_first_member():
    points = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.1], [1.0, 0.05]])
    assert _cosine_average_linkage(points, threshold=0.5) == [[0, 2], [1, 3]]


def test_a_single_point_is_its_own_cluster():
    assert _cosine_average_linkage(np.array([[1.0, 2.0]]), threshold=0.5) == [[0]]


def test_a_zero_vector_never_joins_a_cluster():
    points = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.01]])
    assert _cosine_average_linkage(points, threshold=0.5) == [[0, 2], [1]]


def test_the_cut_is_inclusive():
    # cosine distance exactly 1.0 between orthogonal directions: a cut at 1.0 merges them
    points = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert _cosine_average_linkage(points, threshold=1.0) == [[0, 1]]
    assert _cosine_average_linkage(points, threshold=0.99) == [[0], [1]]


# ---------------------------------------------------------------------------
# Choosing the subject
# ---------------------------------------------------------------------------


def _face(
    side: float,
    embedding: list[float] | None,
    *,
    score: float = 0.9,
    yaw: float | None = None,
) -> Face:
    vector = None if embedding is None else np.asarray(embedding, dtype=np.float32)
    return Face(bbox=(0.0, 0.0, side, side), score=score, embedding=vector, yaw=yaw)


def _normalised_mean(faces: list[Face]) -> np.ndarray:
    stacked = np.vstack([face.embedding for face in faces if face.embedding is not None])
    mean = np.mean(stacked, axis=0).astype(np.float32)
    return (mean / float(np.linalg.norm(mean))).astype(np.float32)


# two well separated identities (cosine distance about 1): a few large faces and more small ones
def _large_faces(*, score: float = 0.9, yaw: float | None = None) -> list[Face]:
    return [
        _face(100, [1.0, 0.02, 0.0], score=score, yaw=yaw),
        _face(100, [1.0, -0.03, 0.01], score=score, yaw=yaw),
    ]


def _small_faces() -> list[Face]:
    return [_face(50, [0.0, 1.0, 0.01 * i]) for i in range(4)]


def test_the_subject_is_the_cluster_with_the_highest_weighted_score():
    # large: 2 x 100*100 x 0.9 = 18000; small: 4 x 50*50 x 0.9 = 9000
    large, small = _large_faces(), _small_faces()
    got = cluster_subject(small + large)
    assert got is not None
    np.testing.assert_array_equal(got, _normalised_mean(large))


def test_the_detection_score_weighs_in():
    # large now scores 2 x 10000 x 0.2 = 4000, below the small faces' 9000
    large, small = _large_faces(score=0.2), _small_faces()
    got = cluster_subject(large + small)
    assert got is not None
    np.testing.assert_array_equal(got, _normalised_mean(small))


@pytest.mark.parametrize(
    ("yaw", "large_wins"),
    [
        (None, True),  # no pose: a frontal weight of 1
        (0.0, True),
        (30.0, True),  # 18000 * 2/3 = 12000
        (-60.0, False),  # 18000 * 1/3 = 6000; the sign of the yaw does not matter
        (90.0, False),  # a pure profile counts for nothing
        (135.0, False),  # and past it the weight stays at zero rather than going negative
    ],
)
def test_turned_away_faces_count_for_less(yaw, large_wins):
    large, small = _large_faces(yaw=yaw), _small_faces()
    got = cluster_subject(large + small)
    assert got is not None
    expected = _normalised_mean(large if large_wins else small)
    np.testing.assert_array_equal(got, expected)


def _three_small_faces() -> list[Face]:
    # 3 x 50*50 x 0.9 = 6750
    return [_face(50, [0.0, 1.0, 0.01 * i]) for i in range(3)]


def test_a_face_turned_past_ninety_degrees_adds_nothing_rather_than_subtracting():
    # frontal: 100*100 x 0.9 = 9000; the turned face would subtract 9000 if its weight went
    # negative, leaving 0 and handing the win to the small faces' 6750
    frontal = _face(100, [1.0, 0.02, 0.0])
    turned = _face(100, [1.0, -0.03, 0.01], yaw=180.0)
    small = _three_small_faces()
    got = cluster_subject([frontal, turned, *small])
    assert got is not None
    np.testing.assert_array_equal(got, _normalised_mean([frontal, turned]))


def test_an_inverted_box_has_no_area_rather_than_a_negative_one():
    upright = _face(100, [1.0, 0.02, 0.0])
    inverted = Face(
        bbox=(100.0, 0.0, 0.0, 100.0),
        score=0.9,
        embedding=np.asarray([1.0, -0.03, 0.01], dtype=np.float32),
    )
    small = _three_small_faces()
    got = cluster_subject([upright, inverted, *small])
    assert got is not None
    np.testing.assert_array_equal(got, _normalised_mean([upright, inverted]))


def test_equal_scores_keep_the_cluster_seen_first():
    left = [_face(10, [1.0, 0.0]), _face(10, [1.0, 0.01])]
    right = [_face(10, [0.0, 1.0]), _face(10, [0.01, 1.0])]
    first = cluster_subject(right + left)
    assert first is not None
    np.testing.assert_array_equal(first, _normalised_mean(right))
    second = cluster_subject(left + right)
    assert second is not None
    np.testing.assert_array_equal(second, _normalised_mean(left))


def test_faces_without_an_embedding_are_left_out():
    huge_unembedded = _face(1000, None)
    small = _small_faces()
    got = cluster_subject([huge_unembedded, *small])
    assert got is not None
    np.testing.assert_array_equal(got, _normalised_mean(small))


def test_the_threshold_is_a_parameter():
    large, small = _large_faces(), _small_faces()
    merged = cluster_subject(large + small, threshold=1.5)
    assert merged is not None
    np.testing.assert_array_equal(merged, _normalised_mean(large + small))


def test_one_face_gives_its_own_embedding_normalised():
    got = cluster_subject([_face(10, [3.0, 4.0])])
    assert got is not None
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.array([0.6, 0.8], dtype=np.float32))


def test_one_face_with_a_zero_embedding_gives_it_back_unchanged():
    got = cluster_subject([_face(10, [0.0, 0.0])])
    assert got is not None
    np.testing.assert_array_equal(got, np.zeros(2, dtype=np.float32))


def test_the_input_embedding_is_not_modified():
    embedding = np.array([3.0, 4.0], dtype=np.float32)
    face = Face(bbox=(0.0, 0.0, 1.0, 1.0), score=0.9, embedding=embedding)
    cluster_subject([face])
    np.testing.assert_array_equal(embedding, np.array([3.0, 4.0], dtype=np.float32))


def test_no_embedded_faces_give_no_subject():
    assert cluster_subject([]) is None
    assert cluster_subject([_face(10, None), _face(20, None)]) is None
