"""The canonical records, and the geometry every later metric is built on.

IoU gets more attention here than its three lines deserve, for one reason: it is
the single arithmetic operation the entire evaluation rests on, and a subtly
wrong one does not crash. It produces plausible metrics that are quietly wrong,
which is the worst failure mode available to this project. Examples pin the
values; properties cover the cases nobody thinks to write down.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ape.records import Box2D, Detection, Difficulty, GroundTruth

# Bounded and quantised so the assertions are about geometry rather than about
# floating point behaviour at 1e300.
coord = st.floats(min_value=-1000, max_value=1000, allow_nan=False,
                  allow_infinity=False, width=32)


@st.composite
def boxes(draw: st.DrawFn) -> Box2D:
    x1, y1 = draw(coord), draw(coord)
    return Box2D(x1, y1, x1 + draw(st.floats(0, 500, width=32)),
                 y1 + draw(st.floats(0, 500, width=32)))


# --- geometry ----------------------------------------------------------------
def test_a_box_reports_its_extent() -> None:
    box = Box2D(10.0, 20.0, 40.0, 60.0)

    assert box.width == pytest.approx(30.0)
    assert box.height == pytest.approx(40.0)
    assert box.area == pytest.approx(1200.0)


def test_an_inverted_box_has_no_extent_rather_than_a_negative_one() -> None:
    """A negative area would sail straight through IoU and produce a union
    smaller than its parts, so degenerate input is clamped at the source."""
    box = Box2D(40.0, 60.0, 10.0, 20.0)

    assert box.width == 0.0
    assert box.height == 0.0
    assert box.area == 0.0


@pytest.mark.parametrize("x,y,inside", [
    (25.0, 40.0, True),
    (10.0, 20.0, True),    # the corner counts
    (40.0, 60.0, True),
    (9.9, 40.0, False),
    (25.0, 60.1, False),
])
def test_containment_includes_the_boundary(x: float, y: float,
                                           inside: bool) -> None:
    assert Box2D(10.0, 20.0, 40.0, 60.0).contains(x, y) is inside


# --- IoU, by example ---------------------------------------------------------
def test_identical_boxes_score_one() -> None:
    box = Box2D(0.0, 0.0, 10.0, 10.0)

    assert box.iou(box) == pytest.approx(1.0)


def test_disjoint_boxes_score_zero() -> None:
    assert Box2D(0, 0, 10, 10).iou(Box2D(20, 20, 30, 30)) == pytest.approx(0.0)


def test_boxes_touching_at_an_edge_score_zero() -> None:
    """The boundary case that decides whether a detector just misses or just
    hits, and the one an off-by-one in the intersection would flip."""
    assert Box2D(0, 0, 10, 10).iou(Box2D(10, 0, 20, 10)) == pytest.approx(0.0)


def test_half_overlap_scores_one_third() -> None:
    """Worked by hand: intersection 50, union 150, so 1/3.

    A hand-computed value rather than a round number, because 0.5 and 1.0 are
    the two answers a broken implementation is most likely to return by accident.
    """
    assert Box2D(0, 0, 10, 10).iou(Box2D(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_a_box_fully_inside_another_scores_the_area_ratio() -> None:
    outer, inner = Box2D(0, 0, 10, 10), Box2D(2, 2, 4, 4)

    assert outer.iou(inner) == pytest.approx(4 / 100)


def test_two_empty_boxes_score_zero_rather_than_dividing_by_zero() -> None:
    empty = Box2D(5.0, 5.0, 5.0, 5.0)

    assert empty.iou(empty) == 0.0


# --- IoU, by property --------------------------------------------------------
@given(a=boxes(), b=boxes())
def test_iou_is_symmetric(a: Box2D, b: Box2D) -> None:
    """Asymmetry would mean the match depends on which argument came first,
    and the matcher would then quietly favour whichever order it happened to
    iterate in."""
    assert a.iou(b) == pytest.approx(b.iou(a), abs=1e-9)


@given(a=boxes(), b=boxes())
def test_iou_stays_within_zero_and_one(a: Box2D, b: Box2D) -> None:
    assert 0.0 <= a.iou(b) <= 1.0 + 1e-9


@given(a=boxes())
def test_a_box_matches_itself_perfectly_unless_it_is_empty(a: Box2D) -> None:
    expected = 1.0 if a.area > 0 else 0.0

    assert a.iou(a) == pytest.approx(expected)


@given(a=boxes(), b=boxes(), gap=st.floats(1.0, 500.0, width=32))
def test_boxes_moved_clear_of_each_other_stop_overlapping(a: Box2D, b: Box2D,
                                                          gap: float) -> None:
    """The shift is derived from the boxes, not picked as a large constant.

    A fixed shift does not separate anything: hypothesis found a pair where
    1001 pixels still left them overlapping, because both the positions and the
    widths are free. Sliding b's left edge past a's right edge is the condition
    that actually holds.
    """
    shift = (a.x2 - b.x1) + gap
    moved = Box2D(b.x1 + shift, b.y1, b.x2 + shift, b.y2)

    assert moved.x1 > a.x2
    assert a.iou(moved) == pytest.approx(0.0)


# --- records -----------------------------------------------------------------
def test_records_are_frozen_so_nothing_downstream_can_rewrite_the_evidence() -> None:
    """Metrics, slicing and the report all read the same objects. A mutable
    record would let a later stage change what an earlier one measured, and the
    report would still look consistent."""
    gt = GroundTruth(frame_id="000000", label="Car", box=Box2D(0, 0, 10, 10),
                     occlusion=0, truncation=0.0)
    det = Detection(frame_id="000000", label="Car", box=Box2D(0, 0, 10, 10),
                    score=0.9)

    with pytest.raises(AttributeError):
        gt.label = "Pedestrian"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        det.score = 1.0  # type: ignore[misc]


def test_a_dataset_without_3d_still_produces_a_valid_record() -> None:
    """The 3D fields are optional on purpose, so an adapter for a dataset with
    no 3D annotation needs no special case anywhere downstream."""
    gt = GroundTruth(frame_id="f", label="Car", box=Box2D(0, 0, 100, 100),
                     occlusion=0, truncation=0.0)

    assert gt.distance_m is None
    assert gt.location_cam is None
    assert gt.difficulty is Difficulty.EASY
