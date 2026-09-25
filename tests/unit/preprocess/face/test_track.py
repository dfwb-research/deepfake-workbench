"""Choosing one face per frame and following it across a clip.

The rules pinned here are the earlier face pipeline's, so that a store built with this framework
picks the same face on every frame as a store built before it: the largest face first, then the
face overlapping the last chosen one most, back to the largest when nothing overlaps enough,
frames without a usable face simply dropped, and (for identity-guided selection) the face most
similar to the clip's subject.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dfwb.core.errors import ConfigError
from dfwb.core.records.local import TrackSpec
from dfwb.preprocess.face.track import TrackResult, select_track
from dfwb.preprocess.face.types import Face

TRACK = TrackSpec(iou=0.3, strategy="largest-then-iou")
IDENTITY = TrackSpec(iou=0.3, strategy="identity-cluster")


def _select(*args: object, **kwargs: object) -> list[tuple[int, Face | None, str | None]]:
    """The per-frame choices alone, for the tests that are not about identity switches."""
    return select_track(*args, **kwargs).frames


def _face(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    score: float = 0.9,
    embedding: list[float] | None = None,
) -> Face:
    vector = None if embedding is None else np.asarray(embedding, dtype=np.float32)
    return Face(bbox=(x1, y1, x2, y2), score=score, embedding=vector)


def _area(face: Face) -> float:
    x1, y1, x2, y2 = face.bbox
    return (x2 - x1) * (y2 - y1)


def _unit(*values: float) -> list[float]:
    vector = np.asarray(values, dtype=np.float64)
    return list(vector / np.linalg.norm(vector))


# ---------------------------------------------------------------------------
# Largest face, then IoU continuity
# ---------------------------------------------------------------------------


def test_first_frame_takes_the_largest_face():
    small = _face(0, 0, 10, 10)
    large = _face(50, 50, 90, 90)
    medium = _face(100, 100, 120, 120)
    assert _select([(0, [small, large, medium])], TRACK, min_score=0.5) == [(0, large, None)]


def test_equally_large_faces_keep_the_first_listed():
    first = _face(0, 0, 10, 10)
    second = _face(50, 50, 60, 60)
    [(_, chosen, _)] = _select([(0, [first, second])], TRACK, min_score=0.5)
    assert chosen is first


def test_iou_continuity_follows_one_face_while_two_faces_cross():
    # Face A moves right and shrinks slowly; face B moves left. A is the larger face on the first
    # frame, so it is the one chosen; from the fourth frame on B is the larger one, and the two
    # cross, so only overlap with the previous choice can keep following A.
    per_frame = []
    tracked = []
    for index in range(9):
        side_a = 44 - index
        face_a = _face(10 * index, 0, 10 * index + side_a, side_a)
        face_b = _face(80 - 10 * index, 5, 80 - 10 * index + 42, 47)
        if index >= 3:
            assert _area(face_b) > _area(face_a)
        # detectors list faces in no particular order
        faces = [face_a, face_b] if index % 2 == 0 else [face_b, face_a]
        per_frame.append((index, faces))
        tracked.append(face_a)

    result = _select(per_frame, TRACK, min_score=0.5)

    assert [index for index, _, _ in result] == list(range(9))
    assert all(reason is None for _, _, reason in result)
    assert all(chosen is face for (_, chosen, _), face in zip(result, tracked, strict=True))


def test_falls_back_to_the_largest_when_the_best_overlap_is_below_the_threshold():
    start = _face(0, 0, 10, 10)
    elsewhere_small = _face(100, 100, 110, 110)
    elsewhere_large = _face(200, 200, 260, 260)
    result = _select([(0, [start]), (1, [elsewhere_small, elsewhere_large])], TRACK, min_score=0.5)
    assert result == [(0, start, None), (1, elsewhere_large, None)]


def test_the_fallback_threshold_comes_from_the_spec():
    spec = TrackSpec(iou=0.5, strategy="largest-then-iou")
    start = _face(0, 0, 10, 10)
    # overlap 100, union 200: an IoU of exactly 0.5, which is not below the threshold
    at_threshold = _face(0, 0, 10, 20)
    # overlap 90, union 210: just under 0.5
    just_below = _face(1, 0, 11, 20)
    bigger = _face(300, 300, 400, 400)

    kept = _select([(0, [start]), (1, [bigger, at_threshold])], spec, min_score=0.5)
    assert kept[1][1] is at_threshold

    dropped = _select([(0, [start]), (1, [just_below, bigger])], spec, min_score=0.5)
    assert dropped[1][1] is bigger

    # the same frames under the usual 0.3 threshold keep following the overlapping face
    loose = _select([(0, [start]), (1, [just_below, bigger])], TRACK, min_score=0.5)
    assert loose[1][1] is just_below


def test_equal_overlaps_keep_the_first_listed_face():
    start = _face(10, 10, 20, 20)
    left = _face(5, 10, 15, 20)
    right = _face(15, 10, 25, 20)
    result = _select([(0, [start]), (1, [left, right])], TRACK, min_score=0.5)
    assert result[1][1] is left
    result = _select([(0, [start]), (1, [right, left])], TRACK, min_score=0.5)
    assert result[1][1] is right


def test_a_degenerate_previous_box_does_not_divide_by_zero():
    # a zero-area box overlapping a zero-area box has a zero union; that counts as no overlap
    point = _face(5, 5, 5, 5)
    same_point = _face(5, 5, 5, 5)
    result = _select([(0, [point]), (1, [same_point])], TRACK, min_score=0.5)
    assert result == [(0, point, None), (1, same_point, None)]


# ---------------------------------------------------------------------------
# Failed frames
# ---------------------------------------------------------------------------


def test_missing_faces_are_dropped_and_tracking_resumes():
    first = _face(0, 0, 40, 40)
    bystander = _face(200, 0, 220, 20)
    # after the gap the tracked face is back near where it was, next to a larger stranger;
    # continuity with the last good face (not the largest face) decides
    returned = _face(4, 0, 44, 40)
    stranger = _face(300, 0, 400, 100)
    per_frame = [
        (0, [first, bystander]),
        (1, []),
        (2, [_face(0, 0, 40, 40, score=0.2)]),
        (3, [stranger, returned]),
    ]

    result = _select(per_frame, TRACK, min_score=0.5)

    assert result == [
        (0, first, None),
        (1, None, "no-face"),
        (2, None, "low-score"),
        (3, returned, None),
    ]


def test_a_failed_frame_is_never_filled_in():
    per_frame = [(0, [_face(0, 0, 10, 10)]), (1, []), (2, [_face(2, 0, 12, 10)])]
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.7)
    result = _select(per_frame, spec, min_score=0.5)
    assert result[1] == (1, None, "no-face")


# ---------------------------------------------------------------------------
# min_score
# ---------------------------------------------------------------------------


def test_faces_below_min_score_count_as_absent():
    large_but_unsure = _face(0, 0, 100, 100, score=0.49)
    small_and_sure = _face(200, 200, 210, 210, score=0.95)
    result = _select([(0, [large_but_unsure, small_and_sure])], TRACK, min_score=0.5)
    assert result == [(0, small_and_sure, None)]


def test_a_score_equal_to_min_score_is_kept():
    face = _face(0, 0, 10, 10, score=0.5)
    assert _select([(0, [face])], TRACK, min_score=0.5) == [(0, face, None)]


def test_a_low_score_face_cannot_hold_the_track():
    start = _face(0, 0, 40, 40)
    # exactly where the tracked face was, but too unsure to count
    ghost = _face(0, 0, 40, 40, score=0.3)
    elsewhere = _face(300, 300, 320, 320)
    result = _select([(0, [start]), (1, [ghost, elsewhere])], TRACK, min_score=0.5)
    assert result[1] == (1, elsewhere, None)


def test_a_frame_whose_faces_are_all_below_min_score_fails_as_low_score():
    faces = [_face(0, 0, 10, 10, score=0.1), _face(20, 20, 40, 40, score=0.4)]
    assert _select([(7, faces)], TRACK, min_score=0.5) == [(7, None, "low-score")]


# ---------------------------------------------------------------------------
# EMA smoothing of the reported box
# ---------------------------------------------------------------------------


def _chain(
    boxes: list[tuple[int, tuple[float, float, float, float]]],
) -> list[tuple[int, list[Face]]]:
    return [(index, [_face(*box)]) for index, box in boxes]


def test_without_ema_the_raw_box_is_reported_unchanged():
    per_frame = _chain([(0, (0, 0, 10, 10)), (1, (1, 0, 11, 10)), (2, (2.5, 0, 12.5, 10))])
    result = _select(per_frame, TRACK, min_score=0.5)
    for (_, faces), (_, chosen, _) in zip(per_frame, result, strict=True):
        assert chosen is faces[0]


def test_ema_blends_the_raw_box_with_the_previous_smoothed_box():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.7)
    raw = [(0.0, 0.0, 10.0, 10.0), (1.3, 0.7, 11.1, 10.9), (2.6, 1.9, 12.4, 11.3)]
    per_frame = _chain([(0, raw[0]), (1, raw[1]), (2, raw[2])])

    result = _select(per_frame, spec, min_score=0.5)

    # written with the literal constants 0.7 and 0.3 on purpose: 1 - 0.7 is not exactly 0.3 in
    # floating point, and the smoothed box has to match the earlier pipeline's to the last bit
    expected_1 = tuple(0.7 * raw[1][i] + 0.3 * raw[0][i] for i in range(4))
    expected_2 = tuple(0.7 * raw[2][i] + 0.3 * expected_1[i] for i in range(4))
    assert result[0][1] is not None
    assert result[0][1].bbox == raw[0]
    assert result[1][1] is not None
    assert result[1][1].bbox == expected_1
    assert result[2][1] is not None
    assert result[2][1].bbox == expected_2


def test_ema_uses_the_configured_weight():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.25)
    per_frame = _chain([(0, (0, 0, 8, 8)), (1, (4, 0, 12, 8))])
    result = _select(per_frame, spec, min_score=0.5)
    assert result[1][1] is not None
    assert result[1][1].bbox == (1.0, 0.0, 9.0, 8.0)


def test_ema_keeps_everything_but_the_box():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.5)
    raw = Face(
        bbox=(2.0, 2.0, 12.0, 12.0),
        score=0.8,
        landmarks5=((3.0, 4.0),) * 5,
        embedding=np.ones(3, dtype=np.float32),
        yaw=12.5,
    )
    result = _select([(0, [_face(0, 0, 10, 10)]), (1, [raw])], spec, min_score=0.5)
    smoothed = result[1][1]
    assert smoothed is not None
    assert smoothed.bbox == (1.0, 1.0, 11.0, 11.0)
    assert (smoothed.score, smoothed.landmarks5, smoothed.yaw) == (0.8, raw.landmarks5, 12.5)
    assert smoothed.embedding is raw.embedding


def test_ema_restarts_after_a_gap_of_more_than_two_frames():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.5)
    per_frame = _chain(
        [
            (0, (0, 0, 10, 10)),
            (2, (2, 0, 12, 10)),  # gap 2: still smoothed
            (5, (4, 0, 14, 10)),  # gap 3: starts afresh from the raw box
            (6, (6, 0, 16, 10)),  # gap 1: smoothed again, from the restart
        ]
    )
    result = _select(per_frame, spec, min_score=0.5)
    boxes = [chosen.bbox for _, chosen, _ in result if chosen is not None]
    assert boxes == [
        (0, 0, 10, 10),
        (1.0, 0.0, 11.0, 10.0),
        (4, 0, 14, 10),
        (5.0, 0.0, 15.0, 10.0),
    ]


def test_ema_gap_is_measured_from_the_last_good_frame():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.5)
    per_frame = [
        (0, [_face(0, 0, 10, 10)]),
        (1, []),
        (2, [_face(2, 0, 12, 10)]),  # two after the last good frame: smoothed
        (3, []),
        (4, [_face(0, 0, 3, 3, score=0.1)]),
        (5, [_face(4, 0, 14, 10)]),  # three after the last good frame: restarted
    ]
    result = _select(per_frame, spec, min_score=0.5)
    boxes = {index: chosen.bbox for index, chosen, _ in result if chosen is not None}
    assert boxes == {0: (0, 0, 10, 10), 2: (1.0, 0.0, 11.0, 10.0), 5: (4, 0, 14, 10)}


def test_overlap_is_always_measured_against_the_raw_box():
    # After the jump on frame 1, the raw box sits at x 8..18 while the smoothed box sits at
    # 5.6..15.6. On frame 2, `raw_side` overlaps the raw box more and `smoothed_side` overlaps the
    # smoothed box more; the raw box must decide.
    spec = TrackSpec(iou=0.3, strategy="largest-then-iou", ema=0.7)
    raw_side = _face(10, 0, 20, 10)
    smoothed_side = _face(4, 0, 14, 10)
    per_frame = [
        (0, [_face(0, 0, 10, 10)]),
        (1, [_face(8, 0, 18, 10)]),
        (2, [smoothed_side, raw_side]),
    ]
    result = _select(per_frame, spec, min_score=0.5)
    chosen = result[2][1]
    assert chosen is not None
    assert chosen.bbox == tuple(
        0.7 * raw_side.bbox[i] + 0.3 * result[1][1].bbox[i] for i in range(4)
    )


# ---------------------------------------------------------------------------
# Identity-guided selection
# ---------------------------------------------------------------------------


def test_subject_picks_the_most_similar_face_over_the_largest():
    subject = np.asarray(_unit(1, 0, 0), dtype=np.float32)
    stranger = _face(0, 0, 100, 100, embedding=_unit(0, 1, 0))
    target = _face(200, 200, 220, 220, embedding=_unit(0.9, 0.1, 0))
    result = _select([(0, [stranger, target])], IDENTITY, min_score=0.5, subject=subject)
    assert result == [(0, target, None)]


def test_subject_selection_ignores_overlap_with_the_previous_face():
    subject = np.asarray(_unit(1, 0, 0), dtype=np.float32)
    target_0 = _face(0, 0, 40, 40, embedding=_unit(1, 0.1, 0))
    # on the next frame a stranger sits exactly where the subject was
    stranger = _face(0, 0, 40, 40, embedding=_unit(0, 1, 0))
    target_1 = _face(300, 0, 340, 40, embedding=_unit(1, 0, 0.2))
    per_frame = [(0, [target_0]), (1, [stranger, target_1])]
    result = _select(per_frame, IDENTITY, min_score=0.5, subject=subject)
    assert [chosen for _, chosen, _ in result] == [target_0, target_1]


def test_equally_similar_faces_keep_the_first_listed():
    subject = np.asarray(_unit(1, 0), dtype=np.float32)
    first = _face(0, 0, 10, 10, embedding=_unit(1, 0.2))
    twin = _face(50, 0, 100, 50, embedding=_unit(1, 0.2))
    less_similar = _face(200, 0, 300, 100, embedding=_unit(1, 0.5))
    result = _select([(0, [first, twin, less_similar])], IDENTITY, min_score=0.5, subject=subject)
    assert result[0][1] is first


def test_no_subject_match_when_the_best_similarity_is_below_point_three():
    subject = np.asarray(_unit(1, 0, 0), dtype=np.float32)
    faces = [
        _face(0, 0, 10, 10, embedding=_unit(0.2, 0.98, 0)),
        _face(20, 0, 30, 10, embedding=_unit(0.29, 0.957, 0)),
    ]
    matching = _face(0, 0, 10, 10, embedding=_unit(0.31, 0.95, 0))
    result = _select([(0, faces), (1, [matching])], IDENTITY, min_score=0.5, subject=subject)
    assert result == [(0, None, "no-subject-match"), (1, matching, None)]


def test_faces_without_an_embedding_are_passed_over_when_others_have_one():
    subject = np.asarray(_unit(0, 1), dtype=np.float32)
    unembedded_large = _face(0, 0, 100, 100)
    embedded_small = _face(200, 200, 210, 210, embedding=_unit(0.1, 1))
    result = _select(
        [(0, [unembedded_large, embedded_small])], IDENTITY, min_score=0.5, subject=subject
    )
    assert result == [(0, embedded_small, None)]


def test_subject_selection_falls_back_to_tracking_when_no_face_has_an_embedding():
    subject = np.asarray(_unit(0, 1), dtype=np.float32)
    start = _face(0, 0, 40, 40, embedding=_unit(0, 1))
    near = _face(2, 0, 42, 40)
    large_far = _face(300, 0, 400, 100)
    small_far = _face(500, 0, 510, 10)
    per_frame = [(0, [start]), (1, [large_far, near]), (2, [small_far, large_far])]
    result = _select(per_frame, IDENTITY, min_score=0.5, subject=subject)
    assert [chosen for _, chosen, _ in result] == [start, near, large_far]


def test_low_score_faces_are_removed_before_matching_the_subject():
    subject = np.asarray(_unit(1, 0), dtype=np.float32)
    perfect_but_unsure = _face(0, 0, 10, 10, score=0.2, embedding=_unit(1, 0))
    decent = _face(20, 0, 30, 10, embedding=_unit(1, 1))
    result = _select([(0, [perfect_but_unsure, decent])], IDENTITY, min_score=0.5, subject=subject)
    assert result == [(0, decent, None)]


def test_identity_cluster_without_a_subject_tracks_like_largest_then_iou():
    per_frame = [
        (0, [_face(0, 0, 10, 10), _face(50, 50, 90, 90)]),
        (1, [_face(300, 300, 400, 400), _face(52, 50, 92, 90)]),
        (2, []),
        (3, [_face(0, 0, 5, 5), _face(600, 0, 610, 10)]),
    ]
    assert _select(per_frame, IDENTITY, min_score=0.5) == _select(per_frame, TRACK, min_score=0.5)


def test_a_subject_is_refused_by_the_largest_then_iou_strategy():
    subject = np.asarray(_unit(1, 0), dtype=np.float32)
    with pytest.raises(ValueError, match="identity-cluster"):
        _select([(0, [_face(0, 0, 1, 1)])], TRACK, min_score=0.5, subject=subject)


# ---------------------------------------------------------------------------
# Strategies and inputs
# ---------------------------------------------------------------------------


def test_all_faces_is_not_available_yet():
    spec = TrackSpec(iou=0.3, strategy="all-faces")
    with pytest.raises(ConfigError, match="not available") as caught:
        _select([(0, [_face(0, 0, 1, 1)])], spec, min_score=0.5)
    assert caught.value.exit_code == 2
    assert "largest-then-iou" in caught.value.hint


def test_an_unknown_strategy_is_a_config_error():
    spec = TrackSpec(iou=0.3, strategy="largest-then-iuo")
    with pytest.raises(ConfigError, match="largest-then-iou"):
        _select([], spec, min_score=0.5)


def test_no_frames_give_no_results():
    assert _select([], TRACK, min_score=0.5) == []


def test_frame_indices_must_increase():
    faces = [_face(0, 0, 1, 1)]
    with pytest.raises(ValueError, match="increasing"):
        _select([(3, faces), (3, faces)], TRACK, min_score=0.5)
    with pytest.raises(ValueError, match="increasing"):
        _select([(3, faces), (1, faces)], TRACK, min_score=0.5)


# ---------------------------------------------------------------------------
# Identity switches
# ---------------------------------------------------------------------------


def _crossing_clip(*, jump_at: int | None) -> list[tuple[int, list[Face]]]:
    """Two faces passing each other, A along the top moving right and shrinking, B along the
    bottom moving left; from the fourth frame on B is the larger. They never overlap, so only a
    break in A's own continuity can move the track. With ``jump_at`` set, A leaps across the
    frame at that index: nothing overlaps the last chosen box, and the tracker has to fall back
    to the largest face."""
    per_frame = []
    for index in range(9):
        side_a = 44 - index
        x_a = 10 * index if jump_at is None or index < jump_at else 10 * index + 400
        face_a = _face(x_a, 0, x_a + side_a, side_a)
        face_b = _face(80 - 10 * index, 100, 80 - 10 * index + 42, 142)
        per_frame.append((index, [face_a, face_b]))
    return per_frame


def test_select_track_returns_a_frozen_track_result():
    result = select_track(_crossing_clip(jump_at=None), TRACK, min_score=0.5)
    assert isinstance(result, TrackResult)
    assert len(result.frames) == 9
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.identity_switch = True  # type: ignore[misc]


def test_a_clean_track_reports_no_identity_switch():
    per_frame = _crossing_clip(jump_at=None)
    result = select_track(per_frame, TRACK, min_score=0.5)
    # A is followed on every frame, though B is the larger face for most of the clip
    assert [chosen for _, chosen, _ in result.frames] == [faces[0] for _, faces in per_frame]
    assert result.identity_switch is False


def test_falling_back_to_the_largest_face_mid_clip_is_an_identity_switch():
    per_frame = _crossing_clip(jump_at=5)
    result = select_track(per_frame, TRACK, min_score=0.5)
    chosen = [chosen for _, chosen, _ in result.frames]
    # A is followed up to the jump; then the fallback takes the larger B, and B is followed
    assert chosen == [faces[0] for _, faces in per_frame[:5]] + [
        faces[1] for _, faces in per_frame[5:]
    ]
    assert result.identity_switch is True


def test_a_clip_without_faces_reports_no_identity_switch():
    result = select_track([(index, []) for index in range(5)], TRACK, min_score=0.5)
    assert [reason for _, _, reason in result.frames] == ["no-face"] * 5
    assert result.identity_switch is False


def test_no_frames_report_no_identity_switch():
    assert select_track([], TRACK, min_score=0.5) == TrackResult(frames=[], identity_switch=False)


def test_the_first_chosen_face_is_never_a_switch_even_after_failed_frames():
    # the first face chosen is always the largest one; that is where tracking starts, not a
    # fallback, however many frames failed before it
    per_frame = [(0, []), (1, [_face(0, 0, 1, 1, score=0.1)]), (2, [_face(0, 0, 10, 10)])]
    assert select_track(per_frame, TRACK, min_score=0.5).identity_switch is False


def test_a_fallback_after_a_failed_frame_is_still_a_switch():
    per_frame = [(0, [_face(0, 0, 10, 10)]), (1, []), (2, [_face(500, 500, 510, 510)])]
    assert select_track(per_frame, TRACK, min_score=0.5).identity_switch is True


def test_the_fallback_counts_even_when_it_lands_on_the_only_face():
    # a lone face that jumped: nothing overlaps, so the fallback runs and takes it
    per_frame = [(0, [_face(0, 0, 10, 10)]), (1, [_face(500, 500, 510, 510)])]
    assert select_track(per_frame, TRACK, min_score=0.5).identity_switch is True


def test_following_the_subject_by_similarity_is_never_a_switch():
    subject = np.asarray(_unit(1, 0), dtype=np.float32)
    per_frame = [
        (0, [_face(0, 0, 10, 10, embedding=_unit(1, 0.1))]),
        (1, [_face(500, 500, 510, 510, embedding=_unit(1, 0.2))]),
        (2, [_face(0, 0, 10, 10, embedding=_unit(0, 1))]),  # no match: a failed frame
    ]
    result = select_track(per_frame, IDENTITY, min_score=0.5, subject=subject)
    assert [reason for _, _, reason in result.frames] == [None, None, "no-subject-match"]
    assert result.identity_switch is False


def test_a_subject_clip_falling_back_to_overlap_can_switch():
    # no face on frame 1 carries an embedding, so overlap decides, and nothing overlaps
    subject = np.asarray(_unit(1, 0), dtype=np.float32)
    per_frame = [
        (0, [_face(0, 0, 10, 10, embedding=_unit(1, 0.1))]),
        (1, [_face(500, 500, 510, 510)]),
    ]
    result = select_track(per_frame, IDENTITY, min_score=0.5, subject=subject)
    assert result.identity_switch is True
