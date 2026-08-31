"""The false-positive taxonomy, and the guard that keeps one matcher.

Two things are tested here. The taxonomy itself, each category built from a
frame where only that category can be the answer. And the property that made
the refactor worth doing: `judge_frame` and `assign_frame` must agree, because
the demo scene used to run its own copy of the matching loop and a copy of a
matcher is a copy that drifts.
"""

from __future__ import annotations

import pytest

from ape.classes import EVALUATED, neutral_labels
from ape.match import assign_frame, judge_frame
from ape.metrics import average_precision, false_negative_rate, recall_at_precision
from ape.outcomes import Outcome, classify, classify_frame
from ape.records import Box2D, Detection, GroundTruth

IOU = 0.5
PED = neutral_labels("Pedestrian")


def truth(x1, y1, x2, y2, label="Pedestrian", frame="f"):
    # Boxes are 80 px tall so KITTI's derived difficulty is EASY rather than
    # IGNORED. Nothing here filters on difficulty, but a fixture built entirely
    # out of objects the benchmark discards would be a trap for the next reader.
    return GroundTruth(frame_id=frame, label=label, box=Box2D(x1, y1, x2, y2),
                       occlusion=0, truncation=0.0)


def detection(x1, y1, x2, y2, score=0.9, label="Pedestrian", frame="f"):
    return Detection(frame_id=frame, label=label, box=Box2D(x1, y1, x2, y2),
                     score=score)


def outcomes(detections, ground_truth):
    return [v.outcome for v in classify_frame(detections, ground_truth,
                                              "Pedestrian", PED, IOU,
                                              EVALUATED)]


# ---- the taxonomy ----------------------------------------------------------

def test_an_exact_box_is_a_true_positive():
    assert outcomes([detection(0, 0, 10, 80)], [truth(0, 0, 10, 80)]) == [
        Outcome.TRUE_POSITIVE]


def test_a_second_box_on_a_found_object_is_a_duplicate_not_a_hallucination():
    """The object is real and was found. NMS territory, not a safety problem."""
    found = detection(0, 0, 10, 80, score=0.9)
    again = detection(1, 0, 11, 80, score=0.4)

    assert outcomes([found, again], [truth(0, 0, 10, 80)]) == [
        Outcome.TRUE_POSITIVE, Outcome.DUPLICATE]


def test_a_box_on_a_car_while_evaluating_pedestrians_is_misclassified():
    """Right place, wrong label. Distinct from inventing an object."""
    assert outcomes([detection(100, 100, 140, 180)],
                    [truth(100, 100, 140, 180, label="Car")]) == [
        Outcome.MISCLASSIFIED]


def test_a_loose_box_on_a_real_pedestrian_is_mislocalised():
    """Overlapping, but under the threshold: a regression problem."""
    result = outcomes([detection(0, 0, 10, 80)], [truth(6, 0, 16, 80)])

    assert result == [Outcome.MISLOCALISED]


def test_a_box_on_empty_road_is_a_hallucination():
    """The category a safety argument cares about, and the only one whose
    cause is invisible in the ground truth."""
    assert outcomes([detection(500, 500, 540, 580)], [truth(0, 0, 10, 80)]) == [
        Outcome.HALLUCINATED]


def test_a_box_on_a_neutral_object_is_ignored_rather_than_wrong():
    """KITTI's own rule: Person_sitting is neither a target nor a mistake."""
    assert outcomes([detection(0, 0, 10, 80)],
                    [truth(0, 0, 10, 80, label="Person_sitting")]) == [
        Outcome.IGNORED]


def test_the_categories_are_distinguishable_from_each_other():
    """A taxonomy whose categories all fire together classifies nothing.

    Every failure kind must be reachable and must exclude the others, or the
    breakdown is an expensive way of counting false positives twice.
    """
    ground_truth = [truth(0, 0, 10, 80), truth(100, 100, 140, 180, label="Car")]
    detections = [
        detection(0, 0, 10, 80, score=0.95),        # true positive
        detection(1, 0, 11, 80, score=0.90),        # duplicate of it
        detection(100, 100, 140, 180, score=0.80),  # misclassified car
        detection(200, 0, 210, 80, score=0.70),     # hallucination
    ]

    assert outcomes(detections, ground_truth) == [
        Outcome.TRUE_POSITIVE, Outcome.DUPLICATE,
        Outcome.MISCLASSIFIED, Outcome.HALLUCINATED]


def test_shares_are_over_false_positives_not_over_all_detections():
    """A model that emits twice as many boxes has twice as much of everything,
    so the comparable number is the fraction, and its denominator matters."""
    ground_truth = {"f": [truth(0, 0, 10, 80)]}
    detections = {"f": [detection(0, 0, 10, 80, score=0.95),
                        detection(1, 0, 11, 80, score=0.90),
                        detection(200, 0, 210, 80, score=0.70)]}

    breakdown = classify(detections, ground_truth, "Pedestrian", PED, IOU,
                         EVALUATED)

    assert breakdown.false_positives == 2
    assert breakdown.share(Outcome.DUPLICATE) == pytest.approx(0.5)
    assert breakdown.share(Outcome.HALLUCINATED) == pytest.approx(0.5)
    assert breakdown.counts[Outcome.TRUE_POSITIVE] == 1


# ---- one matcher, not two --------------------------------------------------

CASES = [
    ("nothing at all", [], []),
    ("one clean hit", [detection(0, 0, 10, 80)], [truth(0, 0, 10, 80)]),
    ("a duplicate", [detection(0, 0, 10, 80, score=0.9),
                     detection(1, 0, 11, 80, score=0.4)],
     [truth(0, 0, 10, 80)]),
    ("a neutral object", [detection(0, 0, 10, 80)],
     [truth(0, 0, 10, 80, label="Person_sitting")]),
    ("a missed object", [], [truth(0, 0, 10, 80)]),
    ("score order decides", [detection(0, 0, 10, 80, score=0.3),
                             detection(0, 4, 10, 84, score=0.99)],
     [truth(0, 0, 10, 80)]),
    ("two objects, one box", [detection(0, 0, 10, 80)],
     [truth(0, 0, 10, 80), truth(0, 4, 10, 84)]),
]


@pytest.mark.parametrize("name,detections,ground_truth",
                         CASES, ids=[c[0] for c in CASES])
def test_judge_frame_and_assign_frame_cannot_disagree(name, detections,
                                                      ground_truth):
    """assign_frame is a view of judge_frame, and this is what keeps it one.

    Before this, scripts/render_demo.py carried its own greedy loop. It was
    close to the real one and not identical, so the picture and the reported
    metric could disagree about the same frame with nothing in the repository
    able to notice.
    """
    detailed = judge_frame(detections, ground_truth, "Pedestrian", PED, IOU)
    flat = assign_frame(detections, ground_truth, "Pedestrian", PED, IOU)

    assert detailed.as_assignment().scored == flat.scored
    assert detailed.as_assignment().positives == flat.positives
    assert detailed.as_assignment().ignored == flat.ignored


def test_found_and_missed_partition_the_countable_objects():
    outcome = judge_frame([detection(0, 0, 10, 80)],
                          [truth(0, 0, 10, 80), truth(300, 0, 310, 80)],
                          "Pedestrian", PED, IOU)

    assert len(outcome.found) == 1
    assert len(outcome.missed) == 1
    assert len(outcome.found) + len(outcome.missed) == len(outcome.counts)


def test_best_overlap_on_separates_unseen_from_badly_boxed():
    """The number the demo scene draws on a red box, now from the matcher."""
    near_miss = truth(6, 0, 16, 80)
    nowhere = truth(800, 300, 810, 380)
    outcome = judge_frame([detection(0, 0, 10, 80)], [near_miss, nowhere],
                          "Pedestrian", PED, IOU)

    assert 0.0 < outcome.best_overlap_on(near_miss) < IOU
    assert outcome.best_overlap_on(nowhere) == 0.0


# ---- more than AP ----------------------------------------------------------

def test_false_negative_rate_is_the_complement_of_the_ceiling():
    ground_truth = {"f": [truth(0, 0, 10, 80), truth(300, 0, 310, 80)]}
    detections = {"f": [detection(0, 0, 10, 80)]}
    from ape.match import assign

    curve = average_precision(assign(detections, ground_truth, "Pedestrian",
                                     PED, IOU))

    assert curve.best_recall == pytest.approx(0.5)
    assert false_negative_rate(curve) == pytest.approx(0.5)


def test_false_negative_rate_is_undefined_with_nothing_to_find():
    curve = average_precision(assign_frame([], [], "Pedestrian", PED, IOU))
    assert false_negative_rate(curve) != false_negative_rate(curve)  # nan


def test_recall_at_precision_reports_zero_when_the_target_is_unreachable():
    """A real answer: this detector cannot be operated that precisely."""
    from ape.match import assign

    ground_truth = {"f": [truth(0, 0, 10, 80)]}
    detections = {"f": [detection(500, 500, 540, 580, score=0.99),
                        detection(0, 0, 10, 80, score=0.10)]}
    curve = average_precision(assign(detections, ground_truth, "Pedestrian",
                                     PED, IOU))

    assert recall_at_precision(curve, 0.99) == 0.0
    # The same detector does reach full recall if precision may fall to a half.
    assert recall_at_precision(curve, 0.5) == pytest.approx(1.0)
