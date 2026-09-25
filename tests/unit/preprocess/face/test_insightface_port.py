"""The ported insightface arithmetic: SCRFD decoding, suppression, ArcFace input and alignment.

These pin the pre- and post-processing ported from insightface 0.7.3 (and, for the similarity
estimate that face alignment needs, from scikit-image) against values worked out by hand, against
independent reference implementations, and against the way the original code built the network's
input. The original was handed frames in OpenCV's blue-green-red order and swapped them to
red-green-blue while building that input; the port is handed red-green-blue frames and skips the
swap, and the tests below check that the network receives exactly the same numbers either way.

No model is needed: a stand-in session returns chosen outputs and records the input it was given.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dfwb.preprocess.face._insightface_port import arcface, face_align, scrfd

STRIDES = (8, 16, 32)


class FakeSession:
    """Stands in for an onnxruntime session: fixed input and output metadata, outputs computed
    from the input blob by ``outputs``, and every input blob kept for inspection."""

    def __init__(
        self,
        outputs: Callable[[np.ndarray], list[np.ndarray]] | None,
        *,
        n_outputs: int = 9,
        input_shape: tuple[object, ...] = (1, 3, "?", "?"),
    ) -> None:
        self._outputs = outputs
        self._n_outputs = n_outputs
        self._input_shape = list(input_shape)
        self.blobs: list[np.ndarray] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input.1", shape=self._input_shape)]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name=f"out{i}", shape=[None, None]) for i in range(self._n_outputs)]

    def run(self, names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert names == [f"out{i}" for i in range(self._n_outputs)]
        assert list(feeds) == ["input.1"]
        blob = feeds["input.1"]
        self.blobs.append(blob)
        assert self._outputs is not None
        return self._outputs(blob)


def _empty_outputs(
    size: int, *, strides: tuple[int, ...] = STRIDES, anchors: int = 2
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """All-zero score, box and landmark outputs for a ``size`` x ``size`` input."""
    rows = [(size // stride) ** 2 * anchors for stride in strides]
    scores = [np.zeros((n, 1), np.float32) for n in rows]
    boxes = [np.zeros((n, 4), np.float32) for n in rows]
    points = [np.zeros((n, 10), np.float32) for n in rows]
    return scores, boxes, points


def _row(x: int, y: int, *, size: int, stride: int, anchor: int = 0, anchors: int = 2) -> int:
    """The output row of the anchor at grid cell (x, y) of the ``stride`` feature map."""
    return ((y * (size // stride)) + x) * anchors + anchor


# ---------------------------------------------------------------------------------- SCRFD maths


def test_anchor_centres_walk_the_grid_row_by_row_repeating_each_centre_per_anchor():
    centres = scrfd.anchor_centers(2, 3, 8, 2)
    expected = [
        [0, 0], [0, 0], [8, 0], [8, 0], [16, 0], [16, 0],
        [0, 8], [0, 8], [8, 8], [8, 8], [16, 8], [16, 8],
    ]  # fmt: skip
    assert centres.dtype == np.float32
    np.testing.assert_array_equal(centres, expected)


def test_anchor_centres_with_one_anchor_per_cell():
    np.testing.assert_array_equal(
        scrfd.anchor_centers(2, 2, 16, 1), [[0, 0], [16, 0], [0, 16], [16, 16]]
    )


def test_distance2bbox_subtracts_left_and_top_and_adds_right_and_bottom():
    points = np.array([[10, 20], [0, 0]], np.float32)
    distance = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], np.float32)
    np.testing.assert_array_equal(
        scrfd.distance2bbox(points, distance), [[9, 18, 13, 24], [-5, -6, 7, 8]]
    )


def test_distance2kps_offsets_each_landmark_from_the_anchor_centre():
    points = np.array([[10, 20]], np.float32)
    distance = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], np.float32)
    np.testing.assert_array_equal(
        scrfd.distance2kps(points, distance), [[11, 22, 13, 24, 15, 26, 17, 28, 19, 30]]
    )


def _reference_nms(boxes: list[list[float]], scores: list[float], thresh: float) -> list[int]:
    """Greedy suppression written out plainly: keep the best remaining box, drop every box whose
    overlap with it (counting pixels inclusively, so a box spans ``x2 - x1 + 1`` pixels) is
    above ``thresh``, repeat."""

    def overlap(a: list[float], b: list[float]) -> float:
        width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
        height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
        inter = width * height
        area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
        area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        return inter / (area_a + area_b - inter)

    remaining = sorted(range(len(boxes)), key=lambda i: -scores[i])
    keep: list[int] = []
    while remaining:
        best, *rest = remaining
        keep.append(best)
        remaining = [i for i in rest if overlap(boxes[best], boxes[i]) <= thresh]
    return keep


def _overlaps(boxes: list[list[float]]) -> list[float]:
    values = []
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
            height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]) + 1)
            inter = width * height
            union = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
            values.append(inter / (union - inter))
    return values


_coordinate = st.floats(0, 100, allow_nan=False, width=32)
_extent = st.floats(0, 60, allow_nan=False, width=32)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(_coordinate, _coordinate, _extent, _extent), min_size=1, max_size=25),
    st.data(),
)
def test_nms_matches_a_plain_greedy_reference(raw_boxes, data):
    boxes = [[x, y, x + w, y + h] for x, y, w, h in raw_boxes]
    scores = data.draw(
        st.lists(
            st.floats(0.125, 1, allow_nan=False, width=32),
            min_size=len(boxes),
            max_size=len(boxes),
            unique=True,
        )
    )
    # float32 (the port) and float64 (the reference) may round an overlap sitting right on the
    # threshold to different sides of it; such boxes say nothing about the algorithm.
    assume(all(abs(value - 0.4) > 1e-4 for value in _overlaps(boxes)))
    dets = np.hstack([np.asarray(boxes), np.asarray(scores)[:, None]]).astype(np.float32)
    keep = scrfd.nms(dets, 0.4)
    assert [int(i) for i in keep] == _reference_nms(boxes, scores, 0.4)


def test_nms_keeps_a_box_whose_overlap_equals_the_threshold():
    # Counting pixels inclusively, each box covers 10 x 10 = 100 pixels and they share 5 x 10 = 50,
    # so the overlap is 50 / 150 = 1/3: a box is only dropped when its overlap is above the
    # threshold, so at a threshold of exactly 1/3 the second box survives.
    dets = np.array([[0, 0, 9, 9, 0.9], [5, 0, 14, 9, 0.8]], np.float32)
    assert [int(i) for i in scrfd.nms(dets, 50 / 150)] == [0, 1]
    assert [int(i) for i in scrfd.nms(dets, 0.3)] == [0]


# ---------------------------------------------------------------------------------- SCRFD model


def test_detect_decodes_boxes_and_landmarks_back_into_frame_pixels():
    scores, boxes, points = _empty_outputs(64)
    # Stride 8, cell (2, 1): anchor centre (16, 8). Distances are in stride units.
    first = _row(2, 1, size=64, stride=8)
    scores[0][first] = 0.9
    boxes[0][first] = [1, 1, 1, 1]
    points[0][first] = [0, 0, 1, 0, 0, 1, -1, 0, 0, -1]
    # The same cell's second anchor: a slightly wider box, lower score, suppressed by the first.
    scores[0][first + 1] = 0.8
    boxes[0][first + 1] = [1, 1, 1.25, 1]
    # Stride 16, cell (1, 2): anchor centre (16, 32).
    second = _row(1, 2, size=64, stride=16)
    scores[1][second] = 0.7
    boxes[1][second] = [0.5, 0.5, 0.5, 0.5]
    # Stride 32: below the detection threshold.
    scores[2][0] = 0.3
    session = FakeSession(lambda blob: [*scores, *boxes, *points])
    detector = scrfd.SCRFD(session, input_size=(64, 64))

    # A 64 x 128 frame is halved to 32 x 64 and placed at the top of the 64 x 64 canvas, so
    # every coordinate the network reports is doubled on the way back.
    frame = np.full((64, 128, 3), 200, np.uint8)
    det, kpss = detector.detect(frame)

    np.testing.assert_allclose(det, [[16, 0, 48, 32, 0.9], [16, 48, 48, 80, 0.7]], rtol=1e-6)
    assert det.dtype == np.float32
    assert kpss is not None
    assert kpss.shape == (2, 5, 2)
    np.testing.assert_allclose(kpss[0], [[32, 16], [48, 16], [32, 32], [16, 16], [32, 0]])
    np.testing.assert_allclose(kpss[1], [[32, 64]] * 5)

    (blob,) = session.blobs
    assert blob.shape == (1, 3, 64, 64)
    assert blob.dtype == np.float32
    np.testing.assert_array_equal(blob[:, :, :32, :], (200 - 127.5) / 128)
    np.testing.assert_array_equal(blob[:, :, 32:, :], (0 - 127.5) / 128)


def test_coordinates_are_scaled_back_by_the_truncated_resized_height():
    # A 90 x 161 frame is resized to int(64 * 90 / 161) = 35 rows by 64 columns, so the scale
    # back is 35 / 90 (not 64 / 161, which the truncation makes slightly different).
    scores, boxes, points = _empty_outputs(64)
    row = _row(2, 2, size=64, stride=8)  # anchor centre (16, 16)
    scores[0][row] = 0.9
    boxes[0][row] = [1, 1, 1, 1]
    session = FakeSession(lambda blob: [*scores, *boxes, *points])
    det, _ = scrfd.SCRFD(session, input_size=(64, 64)).detect(np.zeros((90, 161, 3), np.uint8))
    scale = 35 / 90
    np.testing.assert_allclose(det[0, :4], np.array([8, 8, 24, 24]) / scale, rtol=1e-6)


def test_a_portrait_frame_fills_the_left_of_the_canvas():
    scores, boxes, points = _empty_outputs(64)
    session = FakeSession(lambda blob: [*scores, *boxes, *points])
    detector = scrfd.SCRFD(session, input_size=(64, 64))
    det, kpss = detector.detect(np.full((128, 64, 3), 255, np.uint8))
    assert det.shape == (0, 5)
    assert kpss is not None
    assert kpss.shape == (0, 5, 2)
    (blob,) = session.blobs
    np.testing.assert_array_equal(blob[:, :, :, :32], (255 - 127.5) / 128)
    np.testing.assert_array_equal(blob[:, :, :, 32:], (0 - 127.5) / 128)


def test_forward_keeps_scores_equal_to_the_threshold():
    scores, boxes, points = _empty_outputs(64)
    scores[0][0] = 0.5
    scores[0][1] = np.nextafter(np.float32(0.5), np.float32(0))
    session = FakeSession(lambda blob: [*scores, *boxes, *points])
    detector = scrfd.SCRFD(session, input_size=(64, 64))
    kept_scores, kept_boxes, kept_points = detector.forward(np.zeros((64, 64, 3), np.uint8), 0.5)
    assert [len(s) for s in kept_scores] == [1, 0, 0]
    assert [len(b) for b in kept_boxes] == [1, 0, 0]
    assert [len(p) for p in kept_points] == [1, 0, 0]


def _three_faces() -> FakeSession:
    """Three well separated faces on a 64 x 64 input: a large corner face (score 0.9), a small
    centred face (0.8) and a large face low on the right (0.7)."""
    scores, boxes, points = _empty_outputs(64)
    corner = _row(1, 1, size=64, stride=8)  # centre (8, 8), box 0..16
    scores[0][corner] = 0.9
    boxes[0][corner] = [1, 1, 1, 1]
    centre = _row(4, 4, size=64, stride=8)  # centre (32, 32), box 28..36
    scores[0][centre] = 0.8
    boxes[0][centre] = [0.5, 0.5, 0.5, 0.5]
    low_right = _row(3, 3, size=64, stride=16)  # centre (48, 48), box 32..64
    scores[1][low_right] = 0.7
    boxes[1][low_right] = [1, 1, 1, 1]
    return FakeSession(lambda blob: [*scores, *boxes, *points])


def test_detect_returns_every_face_when_max_num_is_zero():
    detector = scrfd.SCRFD(_three_faces(), input_size=(64, 64))
    det, _ = detector.detect(np.zeros((64, 64, 3), np.uint8))
    np.testing.assert_allclose(det[:, :4], [[0, 0, 16, 16], [28, 28, 36, 36], [32, 32, 64, 64]])


def test_max_num_prefers_large_faces_near_the_centre():
    # area - 2 * squared distance from the frame centre (32, 32):
    # corner 256 - 2 * 1152 = -2048, centre 64 - 0 = 64, low right 1024 - 2 * 512 = 0.
    detector = scrfd.SCRFD(_three_faces(), input_size=(64, 64))
    det, kpss = detector.detect(np.zeros((64, 64, 3), np.uint8), max_num=2)
    np.testing.assert_allclose(det[:, :4], [[28, 28, 36, 36], [32, 32, 64, 64]])
    assert kpss is not None
    assert kpss.shape == (2, 5, 2)


def test_max_num_with_the_max_metric_prefers_the_largest_faces():
    detector = scrfd.SCRFD(_three_faces(), input_size=(64, 64))
    det, _ = detector.detect(np.zeros((64, 64, 3), np.uint8), max_num=2, metric="max")
    np.testing.assert_allclose(det[:, :4], [[32, 32, 64, 64], [0, 0, 16, 16]])


@pytest.mark.parametrize(
    ("n_outputs", "fmc", "strides", "anchors", "use_kps"),
    [
        (6, 3, [8, 16, 32], 2, False),
        (9, 3, [8, 16, 32], 2, True),
        (10, 5, [8, 16, 32, 64, 128], 1, False),
        (15, 5, [8, 16, 32, 64, 128], 1, True),
    ],
)
def test_the_number_of_outputs_sets_the_feature_map_layout(
    n_outputs, fmc, strides, anchors, use_kps
):
    detector = scrfd.SCRFD(FakeSession(None, n_outputs=n_outputs), input_size=(640, 640))
    assert detector.fmc == fmc
    assert detector._feat_stride_fpn == strides
    assert detector._num_anchors == anchors
    assert detector.use_kps is use_kps


def test_an_unknown_output_layout_is_refused():
    with pytest.raises(ValueError, match="7 outputs"):
        scrfd.SCRFD(FakeSession(None, n_outputs=7), input_size=(640, 640))


def test_a_model_without_landmarks_returns_none_for_them():
    scores, boxes, _ = _empty_outputs(64)
    row = _row(2, 2, size=64, stride=8)
    scores[0][row] = 0.9
    boxes[0][row] = [1, 1, 1, 1]
    detector = scrfd.SCRFD(
        FakeSession(lambda blob: [*scores, *boxes], n_outputs=6), input_size=(64, 64)
    )
    det, kpss = detector.detect(np.zeros((64, 64, 3), np.uint8))
    assert kpss is None
    np.testing.assert_allclose(det, [[8, 8, 24, 24, 0.9]], rtol=1e-6)


def test_max_num_works_for_a_model_without_landmarks():
    scores, boxes, _ = _empty_outputs(64)
    for x, score in ((1, 0.9), (4, 0.8), (6, 0.7)):
        row = _row(x, 4, size=64, stride=8)
        scores[0][row] = score
        boxes[0][row] = [0.5, 0.5, 0.5, 0.5]
    detector = scrfd.SCRFD(
        FakeSession(lambda blob: [*scores, *boxes], n_outputs=6), input_size=(64, 64)
    )
    det, kpss = detector.detect(np.zeros((64, 64, 3), np.uint8), max_num=1)
    assert kpss is None
    # all three boxes are the same size, so the one nearest the centre (32, 32) wins
    np.testing.assert_allclose(det, [[28, 28, 36, 36, 0.8]], rtol=1e-6)


def test_a_model_with_a_fixed_input_size_keeps_it_and_warns(caplog):
    session = FakeSession(None, input_shape=(1, 3, 320, 480))
    with caplog.at_level(logging.WARNING):
        detector = scrfd.SCRFD(session, input_size=(640, 640))
    assert detector.input_size == (480, 320)
    assert "fixed input size" in caplog.text


def test_anchor_centres_are_computed_once_per_feature_map():
    scores, boxes, points = _empty_outputs(64)
    detector = scrfd.SCRFD(
        FakeSession(lambda blob: [*scores, *boxes, *points]), input_size=(64, 64)
    )
    frame = np.zeros((64, 64, 3), np.uint8)
    detector.detect(frame)
    cached = dict(detector.center_cache)
    assert sorted(cached) == [(2, 2, 32), (4, 4, 16), (8, 8, 8)]
    detector.detect(frame)
    assert all(detector.center_cache[key] is value for key, value in cached.items())


def test_the_anchor_cache_stops_growing_at_one_hundred_entries():
    scores, boxes, points = _empty_outputs(64)
    detector = scrfd.SCRFD(
        FakeSession(lambda blob: [*scores, *boxes, *points]), input_size=(64, 64)
    )
    detector.center_cache.update({(i, i, 0): np.zeros((0, 2), np.float32) for i in range(100)})
    detector.detect(np.zeros((64, 64, 3), np.uint8))
    assert len(detector.center_cache) == 100


def _insightface_detector_blob(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    """The detector input exactly as insightface 0.7.3 built it from an OpenCV (BGR) frame."""
    im_ratio = float(frame_bgr.shape[0]) / frame_bgr.shape[1]
    if im_ratio > 1.0:
        new_height = size
        new_width = int(new_height / im_ratio)
    else:
        new_width = size
        new_height = int(new_width * im_ratio)
    resized = cv2.resize(frame_bgr, (new_width, new_height))
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:new_height, :new_width, :] = resized
    return cv2.dnn.blobFromImage(
        canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True
    )


@pytest.mark.parametrize("shape", [(90, 160, 3), (160, 90, 3), (77, 77, 3)])
def test_the_detector_sees_what_insightface_gave_it_from_the_same_frame_in_bgr(shape):
    rgb = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    scores, boxes, points = _empty_outputs(64)
    session = FakeSession(lambda blob: [*scores, *boxes, *points])
    scrfd.SCRFD(session, input_size=(64, 64)).detect(rgb)
    expected = _insightface_detector_blob(np.ascontiguousarray(rgb[..., ::-1]), 64)
    (blob,) = session.blobs
    assert np.array_equal(blob, expected)


# ---------------------------------------------------------------------------------- alignment


def _template() -> np.ndarray:
    """The ArcFace landmark template as insightface held it: a float32 array."""
    return np.array(face_align.arcface_dst, dtype=np.float32)


def _similarity(scale: float, degrees: float, tx: float, ty: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cos, sin = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[cos, -sin, tx], [sin, cos, ty]])


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:, :2].T + matrix[:, 2]


def test_landmarks_already_in_place_give_the_identity():
    matrix = face_align.estimate_norm(_template())
    np.testing.assert_allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-5)


def test_estimate_norm_recovers_a_known_similarity():
    forward = _similarity(0.8, 17.0, 5.0, -3.0)  # maps frame landmarks onto the template
    inverse = np.linalg.inv(np.vstack([forward, [0, 0, 1]]))[:2]
    landmarks = _apply(inverse, _template().astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(face_align.estimate_norm(landmarks), forward, atol=1e-4)


@pytest.mark.parametrize(
    ("image_size", "expected"),
    [
        (112, [[1, 0, 0], [0, 1, 0]]),
        (224, [[2, 0, 0], [0, 2, 0]]),
        (128, [[1, 0, 8], [0, 1, 0]]),  # the 128 template is shifted 8 px right
    ],
)
def test_the_template_scales_with_the_output_size(image_size, expected):
    matrix = face_align.estimate_norm(_template(), image_size)
    np.testing.assert_allclose(matrix, expected, atol=1e-4)


def test_estimate_norm_refuses_other_shapes_and_sizes():
    with pytest.raises(ValueError, match="five"):
        face_align.estimate_norm(np.zeros((4, 2), np.float32))
    with pytest.raises(ValueError, match="multiple of 112 or 128"):
        face_align.estimate_norm(_template(), 100)


def _closed_form_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The least-squares rotation, uniform scale and translation from ``src`` to ``dst``,
    computed independently with complex numbers: dst ~ a * src + b."""
    z = src[:, 0] + 1j * src[:, 1]
    w = dst[:, 0] + 1j * dst[:, 1]
    zc, wc = z - z.mean(), w - w.mean()
    a = np.sum(np.conj(zc) * wc) / np.sum(np.abs(zc) ** 2)
    b = w.mean() - a * z.mean()
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


_point = st.tuples(st.floats(-500, 500, allow_nan=False), st.floats(-500, 500, allow_nan=False))


@settings(max_examples=200, deadline=None)
@given(st.lists(_point, min_size=5, max_size=5), st.lists(_point, min_size=5, max_size=5))
def test_umeyama_matches_the_closed_form_least_squares_similarity(src_points, dst_points):
    src = np.asarray(src_points)
    dst = np.asarray(dst_points)
    assume(src.var(axis=0).sum() > 1.0)
    moment = (dst - dst.mean(0)).T @ (src - src.mean(0))
    assume(np.linalg.svd(moment, compute_uv=False)[-1] > 1e-3)
    matrix = face_align.umeyama(src, dst, True)
    np.testing.assert_allclose(matrix[:2], _closed_form_similarity(src, dst), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(matrix[2], [0, 0, 1])


def test_a_mirrored_face_still_gets_a_rotation_not_a_reflection():
    mirrored = _template().astype(np.float64) * [-1, 1]
    matrix = face_align.umeyama(mirrored, _template().astype(np.float64), True)
    assert np.linalg.det(matrix[:2, :2]) > 0
    np.testing.assert_allclose(
        matrix[:2],
        _closed_form_similarity(mirrored, _template().astype(np.float64)),
        atol=1e-9,
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_collinear_points_are_mapped_along_their_line(sign):
    line = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], np.float64)
    target = line * [sign, 1]
    matrix = face_align.umeyama(line, target, True)
    np.testing.assert_allclose(_apply(matrix[:2], line), target, atol=1e-12)
    assert np.linalg.det(matrix[:2, :2]) > 0


def test_coincident_points_give_no_transform():
    points = np.ones((5, 2))
    assert np.isnan(face_align.umeyama(points, points, True)).all()


def test_without_scale_estimation_the_scale_is_one():
    forward = _similarity(3.0, 30.0, 1.0, 2.0)
    src = _template().astype(np.float64)
    matrix = face_align.umeyama(src, _apply(forward, src), False)
    np.testing.assert_allclose(np.linalg.det(matrix[:2, :2]), 1.0)


def test_norm_crop_warps_with_the_estimated_matrix():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(150, 130, 3), dtype=np.uint8)
    landmarks = (_template() * 0.9 + [10, 20]).astype(np.float32)
    matrix = face_align.estimate_norm(landmarks, 112)
    expected = cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)
    assert np.array_equal(face_align.norm_crop(frame, landmarks), expected)


# ---------------------------------------------------------------------------------- ArcFace


def _recognition_session(vector: np.ndarray | None = None) -> FakeSession:
    output = np.arange(512, dtype=np.float32)[None] if vector is None else vector[None]
    return FakeSession(lambda blob: [output], n_outputs=1, input_shape=("None", 3, 112, 112))


def test_arcface_reads_its_input_size_from_the_model():
    model = arcface.ArcFace(_recognition_session())
    assert model.input_size == (112, 112)
    assert (model.input_mean, model.input_std) == (127.5, 127.5)


def test_arcface_needs_a_model_with_exactly_one_output():
    session = FakeSession(None, n_outputs=2, input_shape=("None", 3, 112, 112))
    with pytest.raises(ValueError, match="one output"):
        arcface.ArcFace(session)


def test_arcface_returns_the_flattened_model_output():
    vector = np.linspace(-1, 1, 512, dtype=np.float32)
    model = arcface.ArcFace(_recognition_session(vector))
    frame = np.zeros((150, 130, 3), np.uint8)
    embedding = model.get(frame, _template())
    assert embedding.shape == (512,)
    np.testing.assert_array_equal(embedding, vector)


def test_arcface_sees_what_insightface_gave_it_from_the_same_frame_in_bgr():
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, size=(150, 130, 3), dtype=np.uint8)
    landmarks = (_template() * 0.9 + [10, 20]).astype(np.float32)
    session = _recognition_session()
    arcface.ArcFace(session).get(rgb, landmarks)

    bgr = np.ascontiguousarray(rgb[..., ::-1])
    aligned = cv2.warpAffine(
        bgr, face_align.estimate_norm(landmarks, 112), (112, 112), borderValue=0.0
    )
    expected = cv2.dnn.blobFromImages(
        [aligned], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )
    (blob,) = session.blobs
    assert blob.shape == (1, 3, 112, 112)
    assert np.array_equal(blob, expected)


def test_get_feat_takes_one_image_or_a_list():
    session = FakeSession(
        lambda blob: [np.zeros((blob.shape[0], 512), np.float32)],
        n_outputs=1,
        input_shape=("None", 3, 112, 112),
    )
    model = arcface.ArcFace(session)
    image = np.zeros((112, 112, 3), np.uint8)
    assert model.get_feat(image).shape == (1, 512)
    assert model.get_feat([image, image]).shape == (2, 512)
