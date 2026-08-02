"""Not seen, versus seen and boxed badly.

The whole project counts a miss as a miss. At IoU 0.5 that conflates two
failures with different fixes and different severities, and reporting them as
one number sends an engineer to work on the wrong half. These tests pin the
split, and in particular pin the cases where it is easy to get backwards: a
detection assigned to a neighbouring object still proves the detector saw
something here, and a box that merely clips a corner does not.
"""

from __future__ import annotations

import pytest

from ape.classes import neutral_labels
from ape.localisation import THRESHOLDS, TOUCHING, diagnose
from ape.records import Box2D, Detection, GroundTruth

CAR = neutral_labels("Car")


def gt(x1: float, y1: float, x2: float, y2: float,
       label: str = "Car") -> GroundTruth:
    return GroundTruth(frame_id="f", label=label, box=Box2D(x1, y1, x2, y2),
                       occlusion=0, truncation=0.0, distance_m=20.0)


def det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9,
        label: str = "Car") -> Detection:
    return Detection(frame_id="f", label=label, box=Box2D(x1, y1, x2, y2),
                     score=score)


# --- the split ---------------------------------------------------------------
def test_a_perfect_box_is_found_and_nothing_is_missed() -> None:
    result = diagnose({"f": [det(0, 0, 100, 100)]}, {"f": [gt(0, 0, 100, 100)]},
                      "Car", CAR)

    assert result.total == 1
    assert result.found_tight == 1
    assert result.missed == 0
    assert result.mislocation_share != result.mislocation_share, (
        "with nothing missed the share is undefined, not zero")


def test_a_box_that_lands_badly_counts_as_mislocated_not_unseen() -> None:
    """IoU 0.39: below the 0.5 needed to score, well above the 0.3 that says
    the detector knew something was there."""
    truth = {"f": [gt(0, 0, 100, 100)]}
    detections = {"f": [det(35, 0, 135, 100)]}

    result = diagnose(detections, truth, "Car", CAR)

    assert result.found_tight == 0
    assert result.mislocated == 1
    assert result.unseen == 0
    assert result.mislocation_share == pytest.approx(1.0)


def test_nothing_anywhere_near_counts_as_unseen() -> None:
    truth = {"f": [gt(0, 0, 100, 100)]}
    detections = {"f": [det(500, 500, 600, 600)]}

    result = diagnose(detections, truth, "Car", CAR)

    assert result.unseen == 1
    assert result.mislocated == 0


def test_a_box_that_merely_clips_a_corner_is_not_a_near_miss() -> None:
    """The guard on the diagnosis. Without a floor, any coincidental sliver of
    overlap would be reported as "the detector saw it", which would make the
    mislocation share meaningless."""
    truth = {"f": [gt(0, 0, 100, 100)]}
    detections = {"f": [det(95, 95, 200, 200)]}

    overlap = detections["f"][0].box.iou(truth["f"][0].box)
    assert overlap < TOUCHING

    assert diagnose(detections, truth, "Car", CAR).unseen == 1


def test_a_detection_claimed_by_a_neighbour_still_proves_the_object_was_seen() -> None:
    """The case that is easy to get backwards.

    Two objects, one detection that scores against the first. The second is
    unclaimed, but a box does overlap it: the detector emitted something there.
    Calling that "unseen" would overstate blindness.
    """
    # Offset by 50 of 100, so IoU against the second object is 5000/15000 =
    # 0.333: over the 0.3 floor. At an offset of 60 it is 0.25 and the object
    # is correctly called unseen, which is what the first version of this test
    # got wrong.
    truth = {"f": [gt(0, 0, 100, 100), gt(50, 0, 150, 100)]}
    detections = {"f": [det(0, 0, 100, 100)]}

    assert detections["f"][0].box.iou(truth["f"][1].box) == pytest.approx(1 / 3)

    result = diagnose(detections, truth, "Car", CAR)

    assert result.found_tight == 1
    assert result.mislocated == 1, "the second object was overlapped, not missed"
    assert result.unseen == 0


def test_only_the_class_being_diagnosed_is_considered() -> None:
    """A pedestrian box sitting on a car does not mean the car was seen."""
    truth = {"f": [gt(0, 0, 100, 100)]}
    detections = {"f": [det(0, 0, 100, 100, label="Pedestrian")]}

    result = diagnose(detections, truth, "Car", CAR)

    assert result.found_tight == 0
    assert result.unseen == 1


def test_objects_outside_a_slice_are_not_diagnosed() -> None:
    """The diagnosis has to compose with slicing, or it could only ever be
    reported for the whole dataset."""
    truth = {"f": [gt(0, 0, 100, 100), gt(0, 0, 20, 20)]}

    result = diagnose({}, truth, "Car", CAR,
                      counts_when=lambda item: item.box.area > 1000)

    assert result.total == 1


def test_frames_with_no_objects_of_the_class_are_skipped_not_counted() -> None:
    result = diagnose({}, {"a": [gt(0, 0, 10, 10, label="Tram")],
                           "b": [gt(0, 0, 100, 100)]}, "Car", CAR)

    assert result.total == 1
    assert result.unseen == 1


# --- the thresholds swept ----------------------------------------------------
def test_the_sweep_includes_kittis_own_car_threshold() -> None:
    """The README carried "not comparable to the KITTI leaderboard" as a
    limitation because everything was reported at 0.5. 0.7 is what KITTI
    scores Car at, so it has to be in the sweep for that to be answerable."""
    assert 0.7 in THRESHOLDS
    assert 0.5 in THRESHOLDS


def test_a_looser_tight_threshold_finds_at_least_as_much() -> None:
    """Monotonicity. If relaxing the requirement found fewer objects, the
    assignment is not doing what it says."""
    truth = {"f": [gt(0, 0, 100, 100), gt(200, 0, 300, 100)]}
    detections = {"f": [det(0, 0, 100, 100), det(230, 0, 330, 100)]}

    strict = diagnose(detections, truth, "Car", CAR, tight=0.7)
    relaxed = diagnose(detections, truth, "Car", CAR, tight=0.3)

    assert relaxed.found_tight >= strict.found_tight


def test_the_counts_always_partition_the_objects() -> None:
    truth = {"f": [gt(0, 0, 100, 100), gt(150, 0, 250, 100),
                   gt(400, 0, 500, 100)]}
    detections = {"f": [det(0, 0, 100, 100), det(185, 0, 285, 100)]}

    result = diagnose(detections, truth, "Car", CAR)

    assert result.found_tight + result.mislocated + result.unseen == result.total
