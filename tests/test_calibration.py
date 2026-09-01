"""Calibration: is the detector's confidence worth believing?

The tests that carry weight here are the ones about the SIGN. Expected
calibration error is symmetric, so a timid detector and a confidently wrong one
score identically, and only one of those is a SOTIF problem. Several tests below
construct exactly that pair and assert the two numbers separate them.

The rest guard arithmetic that is easy to get subtly wrong: empty bins that must
not be silently dropped, a score of exactly 1.0 that must land somewhere, and
ignored detections that must not be counted as failures.
"""

from __future__ import annotations

import pytest

from ape.calibration import (
    DEFAULT_BINS,
    by_slice,
    height_band,
    label_band,
    overconfident_slices,
    reliability,
    reliability_table,
    scored,
)
from ape.match import Judged
from ape.records import Box2D, Detection


def _judged(score: float, hit: bool, *, ignored: bool = False,
            height: float = 50.0, label: str = "Car") -> Judged:
    return Judged(
        detection=Detection(frame_id="f", label=label,
                            box=Box2D(0.0, 0.0, 30.0, height), score=score),
        true_positive=hit, ignored=ignored, matched_index=0 if hit else -1,
        best_free=1.0 if hit else 0.0, best_any=1.0 if hit else 0.0)


# --- what counts -------------------------------------------------------------
def test_ignored_detections_are_not_counted_as_failures() -> None:
    """The matcher neither credits nor penalises them.

    Counting them wrong would manufacture overconfidence out of KITTI's
    labelling convention rather than out of the detector's behaviour.
    """
    items = [_judged(0.9, True), _judged(0.9, False, ignored=True)]
    assert scored(items) == [(0.9, True)]
    assert reliability(items).total == 1


def test_a_perfectly_calibrated_detector_has_no_error() -> None:
    """Nine detections at 0.9 confidence, of which eight or nine are right."""
    items = [_judged(0.9, i < 9) for i in range(10)]
    result = reliability(items)
    assert result.expected_calibration_error == pytest.approx(0.0, abs=1e-9)
    assert result.overconfidence_error == pytest.approx(0.0, abs=1e-9)


# --- the sign, which is the point --------------------------------------------
def _overconfident() -> list[Judged]:
    """Claims 0.9, right 4 times in 10."""
    return [_judged(0.9, i < 4) for i in range(10)]


def _underconfident() -> list[Judged]:
    """Claims 0.4, right 9 times in 10."""
    return [_judged(0.4, i < 9) for i in range(10)]


def test_ece_cannot_tell_timid_from_confidently_wrong() -> None:
    """The limitation that motivates the whole module, asserted rather than
    described. Both detectors are off by 0.5 and ECE says they are equal."""
    over = reliability(_overconfident()).expected_calibration_error
    under = reliability(_underconfident()).expected_calibration_error
    assert over == pytest.approx(under, abs=1e-9)


def test_overconfidence_error_separates_them() -> None:
    """The number to read first. SOTIF cares about one of these and not both."""
    over = reliability(_overconfident()).overconfidence_error
    under = reliability(_underconfident()).overconfidence_error
    assert over > 0.4
    assert under == pytest.approx(0.0, abs=1e-9)


def test_a_bin_knows_which_direction_it_is_wrong_in() -> None:
    result = reliability(_overconfident())
    (bin_,) = result.populated
    assert bin_.overconfident
    assert bin_.gap == pytest.approx(0.5)


def test_an_underconfident_bin_is_not_flagged_overconfident() -> None:
    (bin_,) = reliability(_underconfident()).populated
    assert not bin_.overconfident
    assert bin_.gap < 0


# --- the diagram -------------------------------------------------------------
def test_every_bin_exists_even_when_empty() -> None:
    """A detector that never emits 0.3 to 0.7 must not look measured there."""
    result = reliability([_judged(0.95, True)])
    assert len(result.bins) == DEFAULT_BINS
    assert len(result.populated) == 1


def test_an_empty_bin_is_shown_as_empty_rather_than_dropped() -> None:
    rows = reliability_table(reliability([_judged(0.95, True)]))
    blanks = [r for r in rows[1:] if r[1] == "0"]
    assert len(blanks) == DEFAULT_BINS - 1
    assert all(r[2] == "" for r in blanks), "an empty bin has no precision"


def test_a_confidence_of_exactly_one_lands_in_the_top_bin() -> None:
    """Otherwise it indexes a bin that does not exist."""
    result = reliability([_judged(1.0, True)])
    assert result.populated[0].upper == pytest.approx(1.0)


def test_a_confidence_of_zero_lands_in_the_bottom_bin() -> None:
    result = reliability([_judged(0.0, False)])
    assert result.populated[0].lower == pytest.approx(0.0)


def test_the_bin_count_is_a_choice_a_reader_can_change() -> None:
    assert len(reliability([_judged(0.5, True)], bins=4).bins) == 4


def test_zero_bins_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        reliability([], bins=0)


def test_an_empty_set_reports_zero_rather_than_dividing_by_zero() -> None:
    result = reliability([])
    assert result.expected_calibration_error == 0.0
    assert result.overconfidence_error == 0.0
    assert result.maximum_calibration_error == 0.0
    assert result.worst_bin is None


def test_the_worst_bin_is_the_one_driving_the_maximum() -> None:
    items = [_judged(0.95, False)] + [_judged(0.15, False) for _ in range(50)]
    result = reliability(items)
    worst = result.worst_bin
    assert worst is not None
    assert worst.lower == pytest.approx(0.9)
    assert result.maximum_calibration_error == pytest.approx(abs(worst.gap))


def test_the_maximum_is_unweighted_so_a_small_bad_bin_still_shows() -> None:
    """A threshold selects the high-confidence region, so a tiny badly
    calibrated bin up there matters more than its count suggests."""
    items = [_judged(0.95, False)] + [_judged(0.05, False) for _ in range(99)]
    result = reliability(items)
    assert result.maximum_calibration_error > 0.9
    assert result.expected_calibration_error < 0.15


# --- slicing -----------------------------------------------------------------
def test_slicing_is_by_the_detection_so_false_positives_survive() -> None:
    """The design decision worth arguing with, asserted so it cannot drift.

    Every other slice here is cut on ground truth, which exists only for real
    objects. Cutting calibration that way would drop every false positive, and
    those are exactly where overconfidence does its damage.
    """
    items = [_judged(0.9, False, height=10.0), _judged(0.9, True, height=100.0)]
    slices = by_slice(items, height_band)
    assert sum(r.total for r in slices.values()) == 2


def test_height_bands_separate_near_from_far() -> None:
    near = height_band(_judged(0.9, True, height=100.0))
    far = height_band(_judged(0.9, True, height=10.0))
    assert near != far
    assert "over" in near and "under" in far


@pytest.mark.parametrize("height, expected", [
    (10.0, "under 25px"),
    (30.0, "25 to 40px"),
    (60.0, "40 to 80px"),
    (200.0, "over 80px"),
])
def test_every_height_band_is_reachable(height: float, expected: str) -> None:
    """Including the middle ones, which the near-versus-far test skips over.

    A band nothing can land in is a band that silently never appears in a
    slice table, and the reader would take its absence for a clean result.
    """
    assert height_band(_judged(0.9, True, height=height)) == expected


def test_a_binner_returning_none_drops_a_detection_from_every_slice() -> None:
    """Rather than inventing an "other" bucket that reads as a real band."""
    items = [_judged(0.9, True, label="Car"),
             _judged(0.9, True, label="Cyclist")]
    slices = by_slice(items, lambda j: None if j.detection.label == "Cyclist"
                      else j.detection.label)
    assert set(slices) == {"Car"}


def test_slices_are_measured_independently() -> None:
    items = ([_judged(0.9, False, label="Pedestrian") for _ in range(10)]
             + [_judged(0.9, True, label="Car") for _ in range(10)])
    slices = by_slice(items, label_band)
    assert slices["Pedestrian"].overconfidence_error > 0.8
    assert slices["Car"].overconfidence_error == pytest.approx(0.0, abs=1e-9)


def test_overconfident_slices_names_the_bands_worst_first() -> None:
    """The useful output is WHICH band to distrust, not a single boolean."""
    items = ([_judged(0.9, False, label="Pedestrian") for _ in range(10)]
             + [_judged(0.6, False, label="Cyclist") for _ in range(10)]
             + [_judged(0.9, True, label="Car") for _ in range(10)])
    named = overconfident_slices(by_slice(items, label_band), threshold=0.1)
    assert named == ("Pedestrian", "Cyclist"), "Car is calibrated and must not appear"


def test_a_budget_nothing_breaches_returns_nothing() -> None:
    items = [_judged(0.9, True) for _ in range(10)]
    assert overconfident_slices(by_slice(items, label_band), 0.1) == ()
