"""Matching, the three outcomes, and the arithmetic on top of them.

`test_metrics_against_reference.py` proves this agrees with pycocotools. That is
the strongest evidence available and it is not sufficient on its own, because
agreement tells you the two implementations do the same thing and not that the
thing is what KITTI asked for. The neutrality rules in particular are KITTI's,
not COCO's, and the reference has no opinion about them.

So these are the cases stated in prose in `match.py` and `classes.py`, each
pinned to a number that can be worked out by hand.
"""

from __future__ import annotations

import pytest

from ape.classes import (
    EVALUATED,
    HEADLINE,
    ClassMappingError,
    neutral_labels,
    to_kitti,
)
from ape.match import Assignment, assign, assign_frame, at_difficulty, in_slice
from ape.metrics import average_precision, mean_average_precision, precision_recall
from ape.records import Box2D, Detection, Difficulty, GroundTruth


def gt(label: str, x1: float = 100, y1: float = 100, x2: float = 200,
       y2: float = 200, occlusion: int = 0, truncation: float = 0.0,
       distance: float | None = 20.0) -> GroundTruth:
    return GroundTruth(frame_id="f", label=label, box=Box2D(x1, y1, x2, y2),
                       occlusion=occlusion, truncation=truncation,
                       distance_m=distance)


def det(label: str, score: float, x1: float = 100, y1: float = 100,
        x2: float = 200, y2: float = 200) -> Detection:
    return Detection(frame_id="f", label=label, box=Box2D(x1, y1, x2, y2),
                     score=score)


CAR = frozenset(neutral_labels("Car"))


# --- the three outcomes ------------------------------------------------------
def test_a_detection_on_the_right_object_is_a_true_positive() -> None:
    result = assign_frame([det("Car", 0.9)], [gt("Car")], "Car", CAR, 0.5)

    assert result.scored == [(0.9, True)]
    assert result.positives == 1


def test_a_detection_on_nothing_is_a_false_positive() -> None:
    result = assign_frame([det("Car", 0.9, 900, 900, 950, 950)], [gt("Car")],
                          "Car", CAR, 0.5)

    assert result.scored == [(0.9, False)]


def test_a_detection_on_a_van_is_neither_when_evaluating_cars() -> None:
    """KITTI's own rule, and the reason ignoring exists at all.

    Counting this as a false positive would inflate the error for every
    detector equally, which still looks like a result.
    """
    result = assign_frame([det("Car", 0.9)], [gt("Van")], "Car", CAR, 0.5)

    assert result.scored == []
    assert result.ignored == 1
    assert result.positives == 0


def test_a_detection_on_a_dontcare_region_is_never_a_mistake() -> None:
    result = assign_frame([det("Car", 0.9)], [gt("DontCare")], "Car", CAR, 0.5)

    assert result.scored == []
    assert result.ignored == 1


def test_person_sitting_is_neutral_for_pedestrians_but_not_for_cars() -> None:
    """The trap the class mapping exists to avoid: `person` to `Pedestrian`
    looks obvious and hides this."""
    assert "Person_sitting" in neutral_labels("Pedestrian")
    assert "Person_sitting" not in neutral_labels("Car")


def test_a_class_with_no_neutrality_rule_is_refused() -> None:
    """Adding a class to EVALUATED without deciding what is neutral for it
    would silently treat every other annotation as background."""
    with pytest.raises(ClassMappingError, match="no neutrality rule"):
        neutral_labels("Tram")


# --- matching order ----------------------------------------------------------
def test_the_confident_detection_claims_the_object() -> None:
    """Order is part of the definition.

    Match in file order and a weak box steals the object a strong one would
    have taken, which lowers AP for reasons unrelated to the detector.
    """
    weak = det("Car", 0.30, 105, 105, 205, 205)
    strong = det("Car", 0.95, 100, 100, 200, 200)

    result = assign_frame([weak, strong], [gt("Car")], "Car", CAR, 0.5)

    assert (0.95, True) in result.scored
    assert (0.30, False) in result.scored


def test_two_detections_on_one_object_produce_one_hit_and_one_miss() -> None:
    result = assign_frame([det("Car", 0.9), det("Car", 0.8)], [gt("Car")],
                          "Car", CAR, 0.5)

    assert sorted(result.scored, reverse=True) == [(0.9, True), (0.8, False)]


def test_a_detection_of_the_wrong_class_is_not_considered() -> None:
    result = assign_frame([det("Pedestrian", 0.9)], [gt("Car")], "Car", CAR, 0.5)

    assert result.scored == []
    assert result.positives == 1


# --- slicing -----------------------------------------------------------------
def test_an_object_outside_the_slice_is_tolerated_not_dropped() -> None:
    """The whole point of slicing, and wrong in both directions if done the
    obvious way.

    A far car is not a miss for the near-distance slice, and a detection that
    finds it is not a false positive for being correct.
    """
    near, far = gt("Car", distance=5.0), gt("Car", 400, 100, 500, 200, distance=60.0)
    detections = [det("Car", 0.9), det("Car", 0.8, 400, 100, 500, 200)]

    result = assign_frame(detections, [near, far], "Car", CAR, 0.5,
                          in_slice(lambda g: "near" if (g.distance_m or 0) < 10
                                   else "far", "near"))

    assert result.positives == 1, "only the near car counts"
    assert result.scored == [(0.9, True)]
    assert result.ignored == 1, "the far car was tolerated, not scored"


def test_difficulty_tiers_are_cumulative() -> None:
    """Hard means everything up to and including hard, which is KITTI's rule
    and not an interpretation."""
    easy = gt("Car", 100, 100, 200, 200)                    # 100 px, visible
    hard = gt("Car", 300, 100, 340, 130, occlusion=2)       # 30 px, occluded

    at_easy = assign_frame([], [easy, hard], "Car", CAR, 0.5,
                           at_difficulty(Difficulty.EASY))
    at_hard = assign_frame([], [easy, hard], "Car", CAR, 0.5,
                           at_difficulty(Difficulty.HARD))

    assert at_easy.positives == 1
    assert at_hard.positives == 2


def test_frames_with_no_detections_still_count_their_objects() -> None:
    """Otherwise a detector that returns nothing on the hardest frames is
    rewarded for it. This exact bug reached the first evaluation."""
    truth = {"a": [gt("Car")], "b": [gt("Car")]}

    result = assign({"a": [det("Car", 0.9)]}, truth, "Car", CAR, 0.5)

    assert result.positives == 2
    assert result.scored == [(0.9, True)]


# --- metrics -----------------------------------------------------------------
def test_a_perfect_detector_scores_one() -> None:
    perfect = Assignment(scored=[(0.9, True), (0.8, True)], positives=2)

    assert average_precision(perfect).average_precision == pytest.approx(1.0)


def test_a_detector_that_finds_nothing_scores_zero() -> None:
    blind = Assignment(scored=[], positives=5)
    curve = average_precision(blind)

    assert curve.average_precision == 0.0
    assert curve.best_recall == 0.0


def test_a_class_that_is_absent_has_no_score_rather_than_zero() -> None:
    """NaN, not 0.0. Averaging a zero in would report a detector as worse for
    being evaluated on a split that happens to contain no trams."""
    absent = average_precision(Assignment(scored=[(0.9, False)], positives=0))

    assert absent.average_precision != absent.average_precision


def test_the_mean_skips_absent_classes() -> None:
    curves = {
        "Car": average_precision(Assignment([(0.9, True)], positives=1)),
        "Tram": average_precision(Assignment([], positives=0)),
    }

    assert mean_average_precision(curves) == pytest.approx(1.0)


def test_precision_and_recall_run_in_descending_score() -> None:
    curve = Assignment(scored=[(0.5, True), (0.9, True), (0.7, False)], positives=2)

    precision, recall = precision_recall(curve)

    assert precision == pytest.approx([1.0, 0.5, 2 / 3])
    assert recall == pytest.approx([0.5, 0.5, 1.0])


def test_half_the_objects_found_cleanly_scores_about_half() -> None:
    """Worked by hand: two of four found, both ranked first, so precision is 1
    up to recall 0.5 and 0 beyond. Interpolated over 101 points that is the
    51 points from 0.00 to 0.50, so 51/101."""
    half = Assignment(scored=[(0.9, True), (0.8, True)], positives=4)

    assert average_precision(half).average_precision == pytest.approx(51 / 101)


# --- the class mapping -------------------------------------------------------
@pytest.mark.parametrize("coco,kitti", [
    ("car", "Car"), ("truck", "Truck"), ("bus", "Truck"),
    ("person", "Pedestrian"), ("bicycle", "Cyclist"),
])
def test_the_mapping_is_what_it_says_it_is(coco: str, kitti: str) -> None:
    assert to_kitti(coco) == kitti


def test_an_unmapped_class_is_silence_rather_than_a_guess() -> None:
    """A detector firing on a traffic light is not making a claim about any
    KITTI object. Folding it into Misc would turn correct silence into a false
    positive."""
    assert to_kitti("traffic light") is None
    assert to_kitti("toothbrush") is None


def test_cyclist_is_evaluated_but_kept_out_of_the_headline() -> None:
    """KITTI annotates rider and bicycle as one box, a COCO detector emits two,
    and no mapping makes that comparison fair. Reported, and excluded, and the
    reason stated rather than the number quietly dropped."""
    assert "Cyclist" in EVALUATED
    assert "Cyclist" not in HEADLINE
