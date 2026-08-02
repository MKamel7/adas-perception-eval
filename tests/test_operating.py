"""Operating points: the question average precision cannot answer.

AP integrates over every threshold, which is right for comparing detectors and
useless for shipping one. A vehicle runs at a single threshold, and choosing it
trades missed pedestrians against phantom braking. These tests pin the shape of
that trade and, more importantly, the two answers that are not numbers:

  the ceiling      the most that can be caught at ANY threshold. Below target,
                   no threshold choice helps and the answer is a different
                   sensor.

  unreachable      a target recall that no operating point reaches. Reported as
                   None rather than as the nearest available point, because a
                   nearest match reads as though the target had been met.
"""

from __future__ import annotations

import pytest

from ape.match import Assignment
from ape.operating import best_recall, sweep, threshold_for_recall


def assignment(pattern: str, positives: int | None = None) -> Assignment:
    """`pattern` is hits and misses in descending score order: "TTFT"."""
    scored = [(1.0 - i * 0.01, char == "T") for i, char in enumerate(pattern)]
    hits = pattern.count("T")
    return Assignment(scored=scored, positives=positives if positives is not None
                      else hits)


# --- the curve ---------------------------------------------------------------
def test_a_perfect_detector_reaches_full_recall_at_full_precision() -> None:
    points = sweep(assignment("TTTT"), frames=4)

    assert best_recall(points) == pytest.approx(1.0)
    assert points[-1].precision == pytest.approx(1.0)
    assert points[-1].false_positives == 0


def test_recall_never_decreases_as_the_threshold_drops() -> None:
    """Accepting more detections cannot find fewer objects. A violation would
    mean the sweep is not walking the curve in order."""
    points = sweep(assignment("TFTFTTFF"), frames=8)

    recalls = [p.recall for p in points]
    assert recalls == sorted(recalls)


def test_thresholds_descend_as_recall_climbs() -> None:
    points = sweep(assignment("TFTFTTFF"), frames=8)

    thresholds = [p.threshold for p in points]
    assert thresholds == sorted(thresholds, reverse=True)


def test_the_counts_add_up_at_every_point() -> None:
    """Cheap, and it catches an off-by-one in the cursor that would otherwise
    show up as a slightly wrong precision nobody could spot."""
    positives = 5
    points = sweep(assignment("TTFTFFT", positives=positives), frames=7)

    for point in points:
        assert point.true_positives + point.missed == positives
        accepted = point.true_positives + point.false_positives
        assert point.precision == pytest.approx(point.true_positives / accepted)
        assert point.recall == pytest.approx(point.true_positives / positives)


# --- the ceiling, which is the important one ---------------------------------
def test_a_target_above_the_ceiling_is_unreachable_not_approximated() -> None:
    """The finding this exists to express.

    Pedestrian recall on the real evaluation tops out at 0.671. Returning the
    closest point instead of None would let "we need 90% recall" be answered
    with a threshold, implying the target was met.
    """
    points = sweep(assignment("TTFFFF", positives=10), frames=6)

    assert best_recall(points) == pytest.approx(0.2)
    assert threshold_for_recall(points, 0.9) is None
    assert threshold_for_recall(points, 0.2) is not None


def test_the_loosest_threshold_reaching_a_target_is_the_one_returned() -> None:
    """Several thresholds may reach the target; the highest costs the fewest
    false alarms and is the one anybody would choose."""
    points = sweep(assignment("TTFFFF"), frames=6)
    chosen = threshold_for_recall(points, 0.5)

    assert chosen is not None
    reaching = [p for p in points if p.recall >= 0.5]
    assert chosen.threshold == max(p.threshold for p in reaching)
    assert chosen.false_positives == min(p.false_positives for p in reaching)


def test_an_empty_assignment_has_no_curve_and_no_ceiling() -> None:
    assert sweep(Assignment(), frames=0) == []
    assert best_recall([]) == 0.0
    assert threshold_for_recall([], 0.5) is None


def test_a_class_with_no_objects_has_no_curve() -> None:
    assert sweep(Assignment(scored=[(0.9, False)], positives=0), frames=1) == []


# --- the cost, expressed the way an integrator needs it ----------------------
def test_false_alarms_are_reported_per_frame_not_as_a_ratio() -> None:
    """"Two phantom detections every frame" is a quantity somebody can reason
    about. "Precision 0.571" is not."""
    points = sweep(assignment("TFFFF", positives=1), frames=10)
    loosest = points[-1]

    assert loosest.false_positives == 4
    assert loosest.false_alarms_per_frame == pytest.approx(0.4)
    assert loosest.frames_per_false_alarm == pytest.approx(2.5)


def test_a_clean_operating_point_reports_no_false_alarms() -> None:
    points = sweep(assignment("TT"), frames=100)

    assert points[0].false_alarms_per_frame == 0.0
    assert points[0].frames_per_false_alarm == float("inf")


def test_the_rate_is_undefined_rather_than_zero_with_no_frames() -> None:
    """Zero would read as "no false alarms", which is the opposite of "we do
    not know"."""
    points = sweep(assignment("TF"), frames=0)

    assert points[-1].false_alarms_per_frame != points[-1].false_alarms_per_frame


# --- the trade itself --------------------------------------------------------
def test_loosening_the_threshold_buys_recall_and_costs_false_alarms() -> None:
    """The whole argument in one assertion: there is no free recall."""
    points = sweep(assignment("TFTFTFTF", positives=4), frames=8)
    tight, loose = points[0], points[-1]

    assert loose.recall > tight.recall
    assert loose.false_alarms_per_frame >= tight.false_alarms_per_frame
    assert loose.threshold < tight.threshold


def test_thresholds_are_sampled_where_detections_actually_are() -> None:
    """An even grid over [0, 1] wastes most of its points in ranges no
    detection occupies, and misses the sharp region near the top where the
    interesting trade happens."""
    scores = [0.99, 0.98, 0.97, 0.96, 0.95]
    given = Assignment(scored=[(s, True) for s in scores], positives=5)

    points = sweep(given, frames=5, steps=5)

    assert {p.threshold for p in points} <= set(scores)
    assert len(points) == 5
