"""Is this frame the kind of thing the detector was validated on?

Two tests here carry the argument rather than checking arithmetic:

  * `test_a_joint_excursion_is_caught_where_per_feature_scores_would_miss_it`
    is why this uses Mahalanobis distance instead of a z-score per feature.
    A foggy frame is brighter AND lower contrast AND has fewer edges together;
    scoring each independently treats one moderate joint excursion as three
    unremarkable ones, which is precisely the case that matters.
  * `test_an_uninformative_score_is_reported_as_uninformative` is the honesty
    check. An OOD score nobody has validated is a number, not evidence, and a
    score that cannot rank the degraded frames first has not earned the right
    to gate anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from ape.ood import (
    FEATURE_NAMES,
    agreement,
    fit_envelope,
    triggering_candidates,
)


def _reference(rng: np.random.Generator, n: int = 200) -> np.ndarray:
    """Two strongly correlated features, which is what real image stats do."""
    base = rng.normal(0.0, 1.0, n)
    return np.column_stack([base, base + rng.normal(0.0, 0.05, n)])


# --- the envelope ------------------------------------------------------------
def test_one_reference_frame_is_refused() -> None:
    """One frame has no spread, so every other frame is infinitely far away."""
    with pytest.raises(ValueError, match="at least two reference frames"):
        fit_envelope([[0.0, 0.0]])


def test_a_frame_at_the_centre_scores_about_zero() -> None:
    rng = np.random.default_rng(0)
    envelope = fit_envelope(_reference(rng))
    assert envelope.score(envelope.mean) == pytest.approx(0.0, abs=1e-6)


def test_the_score_grows_with_distance() -> None:
    rng = np.random.default_rng(1)
    envelope = fit_envelope(_reference(rng))
    near = envelope.score(envelope.mean + np.array([1.0, 1.0]))
    far = envelope.score(envelope.mean + np.array([4.0, 4.0]))
    assert far > near


def test_a_joint_excursion_is_caught_where_per_feature_scores_would_miss_it() -> None:
    """The reason this is Mahalanobis and not a z-score per feature.

    Both probes sit two standard deviations out on each feature, so any
    per-feature test scores them identically. One moves ALONG the correlation
    the reference data has and is ordinary; the other moves ACROSS it and is
    the kind of frame that should be flagged.
    """
    rng = np.random.default_rng(2)
    reference = _reference(rng)
    envelope = fit_envelope(reference)
    spread = reference.std(axis=0)

    along = envelope.score(envelope.mean + np.array([2 * spread[0], 2 * spread[1]]))
    across = envelope.score(envelope.mean + np.array([2 * spread[0], -2 * spread[1]]))

    assert across > 10 * along, (
        "a violation of the correlation must score far above a move along it")


def test_a_singular_covariance_does_not_explode() -> None:
    """Two identical features make the sample covariance singular.

    Without the ridge the inverse turns rounding noise into enormous distances,
    so the score becomes unstable exactly when the reference set is small,
    which is when it is most likely to be used.
    """
    duplicated = [[x, x] for x in (0.0, 1.0, 2.0, 3.0)]
    envelope = fit_envelope(duplicated, ridge=1e-6)
    score = envelope.score(np.array([1.5, 1.5]))
    assert np.isfinite(score)


def test_a_bigger_ridge_makes_the_score_more_conservative() -> None:
    """Its effect is stated in the docstring, so it is asserted here."""
    rng = np.random.default_rng(3)
    reference = _reference(rng)
    probe = reference.mean(axis=0) + np.array([1.0, -1.0])
    small = fit_envelope(reference, ridge=1e-9).score(probe)
    large = fit_envelope(reference, ridge=1.0).score(probe)
    assert large < small


def test_scoring_with_the_wrong_number_of_features_is_refused() -> None:
    """A reference fitted on one feature set cannot score another."""
    envelope = fit_envelope([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="expected 2 features"):
        envelope.score(np.array([0.0, 0.0, 0.0]))


def test_the_envelope_records_how_many_frames_it_was_fitted_on() -> None:
    """A score against four frames deserves less trust than one against 400."""
    assert fit_envelope([[0.0, 0.0], [1.0, 1.0], [2.0, 2.5]]).count == 3


# --- validating the score ----------------------------------------------------
def test_a_score_that_predicts_degradation_is_informative() -> None:
    scores = {"a": 0.1, "b": 0.2, "c": 5.0, "d": 6.0}
    performance = {"a": 0.9, "b": 0.85, "c": 0.3, "d": 0.2}
    result = agreement(scores, performance, threshold=1.0)
    assert result.is_informative
    assert result.auc == pytest.approx(1.0)
    assert result.separation > 0.5
    assert (result.normal, result.novel) == (2, 2)


def test_a_perfectly_inverted_score_is_reported_as_uninformative() -> None:
    """High score on the frames that did BEST. The worst possible ranking."""
    scores = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    performance = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    result = agreement(scores, performance, threshold=2.5)
    assert not result.is_informative
    assert result.auc == pytest.approx(0.0)


def test_a_mostly_wrong_score_lands_below_a_coin_toss() -> None:
    """Ranks correctly within each group and wrongly across them, which is
    worse than useless and must be reported as such.

    Two of the six pairs rank correctly: a against b, and c against d. All four
    across-group pairs are wrong, so the score sits at a third rather than at
    zero. Asserting the exact value because a test that only checked "< 0.5"
    would pass on an arithmetic error that moved it anywhere below the line.
    """
    scores = {"a": 5.0, "b": 6.0, "c": 0.1, "d": 0.2}
    performance = {"a": 0.9, "b": 0.85, "c": 0.3, "d": 0.2}
    result = agreement(scores, performance, threshold=1.0)
    assert not result.is_informative
    assert result.auc == pytest.approx(1 / 3)
    assert result.separation < 0


def test_a_score_carrying_no_information_sits_at_a_coin_toss() -> None:
    scores = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
    performance = {"a": 0.9, "b": 0.1, "c": 0.8, "d": 0.2}
    assert agreement(scores, performance, 0.5).auc == pytest.approx(0.5)


def test_frames_with_no_recorded_outcome_are_dropped() -> None:
    """Scoring a frame whose result is unknown would invent one for it."""
    result = agreement({"a": 0.1, "ghost": 9.0}, {"a": 0.9}, threshold=1.0)
    assert (result.normal, result.novel) == (1, 0)


def test_agreement_over_nothing_is_a_coin_toss_not_a_crash() -> None:
    result = agreement({}, {}, threshold=1.0)
    assert result.auc == pytest.approx(0.5)
    assert result.mean_normal_performance == 0.0
    assert result.mean_novel_performance == 0.0


def test_frames_that_all_performed_identically_carry_no_ranking() -> None:
    scores = {"a": 0.1, "b": 9.0}
    performance = {"a": 0.5, "b": 0.5}
    assert agreement(scores, performance, 1.0).auc == pytest.approx(0.5)


# --- what comes out ----------------------------------------------------------
def test_candidates_come_back_furthest_first() -> None:
    scores = {"a": 0.1, "b": 4.0, "c": 9.0}
    assert triggering_candidates(scores, threshold=1.0) == ("c", "b")


def test_nothing_outside_the_envelope_returns_nothing() -> None:
    assert triggering_candidates({"a": 0.1}, threshold=1.0) == ()


def test_the_feature_names_match_what_the_extractor_produces() -> None:
    """A reference fitted on one version must not silently score another."""
    pillow = pytest.importorskip("PIL.Image")

    from ape.ood import image_features

    image = pillow.new("RGB", (32, 32), (120, 130, 140))
    assert len(image_features(image)) == len(FEATURE_NAMES)


def test_a_dark_frame_and_a_bright_frame_produce_different_features() -> None:
    """The extractor has to actually respond to the conditions it names."""
    pillow = pytest.importorskip("PIL.Image")

    from ape.ood import image_features

    dark = image_features(pillow.new("RGB", (32, 32), (10, 10, 10)))
    bright = image_features(pillow.new("RGB", (32, 32), (240, 240, 240)))
    assert dark[0] < bright[0], "mean luma must separate them"


def test_a_night_frame_is_flagged_against_a_daylight_envelope() -> None:
    """The end to end claim, on images rather than on invented vectors."""
    pillow = pytest.importorskip("PIL.Image")

    from ape.ood import image_features

    rng = np.random.default_rng(4)
    daylight = []
    for _ in range(12):
        noise = rng.integers(150, 210, size=(24, 24, 3), dtype=np.uint8)
        daylight.append(image_features(pillow.fromarray(noise, "RGB")))
    envelope = fit_envelope(daylight, ridge=1e-8)

    ordinary = rng.integers(150, 210, size=(24, 24, 3), dtype=np.uint8)
    night = rng.integers(0, 25, size=(24, 24, 3), dtype=np.uint8)

    assert (envelope.score(image_features(pillow.fromarray(night, "RGB")))
            > envelope.score(image_features(pillow.fromarray(ordinary, "RGB"))))
