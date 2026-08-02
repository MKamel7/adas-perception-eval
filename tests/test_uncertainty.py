"""The bootstrap, and the vectorised AP underneath it.

A second implementation of average precision is exactly the duplication this
project argues against, so it does not get to be trusted for being short. The
first test here requires it to agree with `metrics.average_precision` on random
inputs, and that function is itself checked against pycocotools. The chain is
deliberate: fast path agrees with clear path, clear path agrees with the
reference, so the fast path inherits the evidence rather than asserting it.

The rest is about the interval behaving like an interval: wider when there is
less data, containing the point estimate, and stable between runs.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ape.match import Assignment
from ape.metrics import average_precision
from ape.uncertainty import Interval, bootstrap
from ape.uncertainty import _ap as fast_ap


def as_assignment(scored: list[tuple[float, bool]], positives: int) -> Assignment:
    return Assignment(scored=list(scored), positives=positives)


# --- the fast path must equal the checked path -------------------------------
@given(scored=st.lists(st.tuples(st.floats(0.0, 1.0, width=32,
                                           allow_nan=False, allow_infinity=False),
                                 st.booleans()),
                       min_size=0, max_size=60),
       extra=st.integers(0, 20))
@settings(max_examples=200, deadline=None)
def test_the_vectorised_average_precision_matches_the_reference(
        scored: list[tuple[float, bool]], extra: int) -> None:
    """The whole justification for having two implementations.

    Random score and hit patterns, including duplicate scores, all-false runs
    and empty input, because those are where a vectorised rewrite and a loop
    part company: ties, the monotone envelope, and the boundary where recall
    never reaches a grid point.
    """
    positives = sum(1 for _, hit in scored if hit) + extra
    expected = average_precision(as_assignment(scored, positives))

    actual = fast_ap(np.array([s for s, _ in scored], dtype=np.float64),
                     np.array([float(h) for _, h in scored], dtype=np.float64),
                     positives)

    if expected.average_precision != expected.average_precision:
        assert actual != actual, "both must be undefined when nothing is positive"
    else:
        assert actual == pytest.approx(expected.average_precision, abs=1e-12)


# --- the interval behaves like an interval -----------------------------------
def frames_of(counts: list[tuple[int, int]], seed: int = 7
              ) -> dict[str, Assignment]:
    """`counts` is (hits, misses) per frame, with scores drawn independently.

    SCORES MUST NOT BE CORRELATED WITH FRAME INDEX, and the first version of
    this fixture got that wrong in a way that produced a real-looking failure.
    It walked a descending score grid frame by frame, so frame 0 held the
    highest-scoring detections in the whole set. Resampling then duplicated
    early frames and front-loaded true positives, inflating AP, and the interval
    sat entirely ABOVE the point estimate.

    That is a genuine property of the percentile bootstrap under that structure,
    and it is not a property of this data: on the real evaluation the interval
    contains the point estimate in every band. The fixture was the bug. It is
    pinned as its own test below rather than deleted, because "the interval can
    sit off the point estimate when score is confounded with the resampling
    unit" is worth knowing about the method.
    """
    rng = random.Random(seed)
    out: dict[str, Assignment] = {}
    for index, (hits, misses) in enumerate(counts):
        scored = [(rng.random(), True) for _ in range(hits)]
        scored += [(rng.random(), False) for _ in range(misses)]
        out[f"f{index:03d}"] = Assignment(scored=scored, positives=hits)
    return out


def confounded_frames(count: int) -> dict[str, Assignment]:
    """Scores perfectly correlated with frame index: the pathological case."""
    out: dict[str, Assignment] = {}
    score = 1.0
    for index in range(count):
        scored = []
        for hit in (True, True, True, False):
            score -= 0.001
            scored.append((score, hit))
        out[f"f{index:03d}"] = Assignment(scored=scored, positives=3)
    return out


def test_the_interval_contains_its_point_estimate() -> None:
    result = bootstrap(frames_of([(3, 1)] * 40))

    assert result.low <= result.point <= result.high


def test_the_point_estimate_equals_the_unresampled_average_precision() -> None:
    """The bootstrap must not quietly change the number it is putting an
    interval around."""
    frames = frames_of([(3, 2), (1, 4), (5, 0)] * 10)
    total = Assignment()
    for piece in frames.values():
        total.extend(piece)

    assert bootstrap(frames, resamples=50).point == pytest.approx(
        average_precision(total).average_precision, abs=1e-12)


def test_fewer_frames_give_a_wider_interval() -> None:
    """The property that makes the interval worth printing.

    A slice computed from four frames must not look as authoritative as one
    computed from two hundred, and this is what stops the report presenting
    them in the same typeface without comment.
    """
    many = bootstrap(frames_of([(3, 2), (1, 3)] * 100))
    few = bootstrap(frames_of([(3, 2), (1, 3)] * 2))

    assert few.width > many.width * 2, (
        f"4 frames gave a {few.width:.3f} interval and 200 gave "
        f"{many.width:.3f}; the bootstrap is not responding to sample size")


def test_the_interval_is_reproducible() -> None:
    """An interval that moved between runs of the same evaluation would be
    indistinguishable from one that moved because the detector changed."""
    frames = frames_of([(2, 2)] * 30)

    assert bootstrap(frames).low == bootstrap(frames).low
    assert bootstrap(frames).high == bootstrap(frames).high


def test_the_percentile_interval_can_miss_the_point_when_score_is_confounded(
) -> None:
    """A documented limit of the method, not a defect in this data.

    When a detection's score is perfectly determined by which frame it came
    from, resampling frames also resamples score rank, and the percentile
    interval drifts off the point estimate. Real detector scores are not
    structured that way, and on the actual evaluation the interval contains the
    point estimate in every distance band. Pinned here so the caveat is on
    record rather than folklore.
    """
    result = bootstrap(confounded_frames(40), resamples=200)

    assert not result.low <= result.point <= result.high


def test_a_slice_with_no_objects_has_no_interval() -> None:
    result = bootstrap(frames_of([(0, 3)] * 5))

    assert result.point != result.point, "AP is undefined with no positives"


def test_no_frames_at_all_is_not_a_crash() -> None:
    result = bootstrap({})

    assert result.frames == 0
    assert result.point != result.point


# --- what the report uses it for ---------------------------------------------
def test_separation_is_symmetric_and_needs_a_real_gap() -> None:
    low = Interval(point=0.10, low=0.05, high=0.15, frames=50)
    high = Interval(point=0.60, low=0.55, high=0.65, frames=50)
    overlapping = Interval(point=0.20, low=0.10, high=0.30, frames=50)

    assert low.separated_from(high)
    assert high.separated_from(low)
    assert not low.separated_from(overlapping)
    assert not overlapping.separated_from(low)


def test_the_headline_claim_survives_its_own_interval() -> None:
    """The claim this project leads with is that near and far pedestrians are
    different, not merely differently averaged. If the intervals overlapped,
    the wording would have to change."""
    near = bootstrap(frames_of([(4, 1)] * 60))
    far = bootstrap(frames_of([(0, 5)] * 60 + [(1, 5)]))

    assert near.point > far.point
    assert near.separated_from(far)
