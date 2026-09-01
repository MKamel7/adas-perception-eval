"""Is this frame the kind of thing the detector was validated on?

WHY THIS BELONGS IN A SOTIF HARNESS. ISO 21448 is about hazards that arise with
no component failing: the detector works exactly as designed and the WORLD is
outside what it was designed for. Every other measurement here answers "how well
did it do on this data". This one answers the prior question, "is this data the
data we validated against", and a frame that is not is a triggering condition
whether or not the detector happened to get it right.

WHAT THIS IS, stated plainly so nobody reads more into it. It is a
FEATURE-SPACE novelty detector over simple image statistics, fitted on a
reference set and scored by Mahalanobis distance. It is NOT a learned OOD
method, there is no network here and nothing is trained, which is consistent
with the rest of this repository. It will notice fog, night, blur, a blown
exposure and a compression artefact storm. It will NOT notice a semantically
novel object rendered at ordinary brightness and contrast, and that limit is
the interesting half of the honesty: it is exactly the failure a statistics-only
detector cannot see, and saying so is better than letting a low score be read as
"nothing unusual here".

WHY MAHALANOBIS AND NOT A PER-FEATURE Z-SCORE. The features covary: a foggy
frame is brighter AND lower contrast AND has fewer edges together. Scoring each
independently and taking a maximum treats one moderate joint excursion as three
unremarkable ones, which is the case that matters. Mahalanobis measures the
excursion in the correlated space.

THE COVARIANCE IS SHRUNK, deliberately. With a handful of reference frames and
several features the sample covariance is singular or nearly so, and its inverse
then produces enormous distances from rounding noise. A small ridge is added to
the diagonal, the amount is a parameter rather than a constant, and its effect
is that the score becomes CONSERVATIVE rather than unstable.

AN OOD SCORE NOBODY HAS VALIDATED IS A NUMBER, NOT EVIDENCE. `agreement()`
exists for that: it measures whether high-scoring frames actually did worse. A
score that does not separate the degraded frames from the healthy ones has not
earned the right to gate anything, and this module gives a caller the means to
find that out rather than assuming it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

#: Names of the statistics `image_features` produces, in order. Kept beside the
#: extractor so a reference distribution fitted on one version cannot be scored
#: against features from another without the length mismatch being obvious.
FEATURE_NAMES: tuple[str, ...] = (
    "mean_luma", "std_luma", "p05_luma", "p95_luma", "edge_density",
    "colour_spread",
)


def image_features(image: Any) -> np.ndarray:
    """Six cheap statistics that move when the operating conditions move.

    Chosen because each one corresponds to a condition a validation engineer
    would actually name: mean and percentile luma for exposure and night, its
    standard deviation and the 5th to 95th spread for contrast and fog, edge
    density for blur and compression, colour spread for a monochrome or heavily
    tinted scene.

    PIL is imported here rather than at module import, matching `ape.perturb`,
    so that the numeric core of this module stays importable without it.
    """
    from PIL import ImageFilter

    grey = image.convert("L")
    luma = np.asarray(grey, dtype=np.float64) / 255.0
    edges = np.asarray(grey.filter(ImageFilter.FIND_EDGES),
                       dtype=np.float64) / 255.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

    return np.array([
        float(luma.mean()),
        float(luma.std()),
        float(np.percentile(luma, 5)),
        float(np.percentile(luma, 95)),
        # Mean edge magnitude. Blur removes edges; compression adds spurious
        # ones at block boundaries, so this moves in both directions and the
        # Mahalanobis distance does not care which.
        float(edges.mean()),
        # How far apart the channels are on average. Near zero for a greyscale
        # or a strongly tinted frame.
        float(rgb.std(axis=2).mean()),
    ])


@dataclass(frozen=True)
class Envelope:
    """The operating envelope: what the reference frames looked like.

    Frozen because an envelope that changes after frames have been scored
    against it makes those scores incomparable, and nobody would be told.
    """

    mean: np.ndarray
    inverse_covariance: np.ndarray
    count: int
    ridge: float

    def score(self, features: np.ndarray) -> float:
        """Mahalanobis distance from the reference distribution.

        Distance rather than squared distance, so the number is in units of
        standard deviations along the worst direction and a threshold of 3 means
        what a reader expects it to mean.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.shape != self.mean.shape:
            raise ValueError(
                f"expected {self.mean.shape[0]} features, got "
                f"{features.shape[0]}. A reference fitted on one feature set "
                f"cannot score another.")
        delta = features - self.mean
        squared = float(delta @ self.inverse_covariance @ delta)
        # Clamped at zero: a shrunk inverse covariance is positive definite in
        # exact arithmetic, and floating point can still produce a tiny negative
        # for a frame sitting on the mean.
        return float(np.sqrt(max(squared, 0.0)))


def fit_envelope(reference: Iterable[Sequence[float]],
                 ridge: float = 1e-6) -> Envelope:
    """Fit the operating envelope from reference frames.

    At least two frames are required and the requirement is real rather than
    defensive: one frame has no spread, so every other frame is infinitely far
    from it and the score would be meaningless rather than merely uncertain.
    """
    matrix = np.asarray([list(row) for row in reference], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError(
            "an operating envelope needs at least two reference frames; one "
            "frame has no spread to measure an excursion against")

    mean = matrix.mean(axis=0)
    covariance = np.cov(matrix, rowvar=False)
    covariance = np.atleast_2d(covariance)
    # The ridge is what makes this usable on a small reference set. Without it
    # the covariance of six features over ten frames is singular and its
    # inverse turns rounding noise into enormous distances.
    covariance = covariance + ridge * np.eye(covariance.shape[0])
    return Envelope(mean=mean, inverse_covariance=np.linalg.inv(covariance),
                    count=matrix.shape[0], ridge=ridge)


@dataclass(frozen=True)
class Agreement:
    """Does the OOD score actually predict that the detector did worse?

    `separation` is the difference in mean performance between frames the score
    called normal and frames it called novel. Positive means the score is
    earning its keep. `auc` is the probability that a randomly chosen degraded
    frame scores higher than a randomly chosen healthy one, which is 0.5 for a
    score that carries no information at all.
    """

    normal: int
    novel: int
    mean_normal_performance: float
    mean_novel_performance: float
    auc: float

    @property
    def separation(self) -> float:
        return self.mean_normal_performance - self.mean_novel_performance

    @property
    def is_informative(self) -> bool:
        """Better than a coin toss at ranking the degraded frames first.

        Deliberately a low bar. It is the threshold below which a score should
        not be allowed to gate anything, not a standard worth being pleased by.
        """
        return self.auc > 0.5


def agreement(scores: dict[str, float], performance: dict[str, float],
              threshold: float) -> Agreement:
    """Check the score against what actually happened, frame by frame.

    `performance` is any per-frame quality figure where higher is better, so a
    caller can pass recall, an F1 or a per-frame AP without this module needing
    an opinion about which. Frames missing from either mapping are dropped,
    because scoring a frame whose outcome is unknown would quietly invent a
    result for it.
    """
    shared = sorted(set(scores) & set(performance))
    normal = [performance[f] for f in shared if scores[f] <= threshold]
    novel = [performance[f] for f in shared if scores[f] > threshold]

    return Agreement(
        normal=len(normal), novel=len(novel),
        mean_normal_performance=float(np.mean(normal)) if normal else 0.0,
        mean_novel_performance=float(np.mean(novel)) if novel else 0.0,
        auc=_auc(scores, performance, shared),
    )


def _auc(scores: dict[str, float], performance: dict[str, float],
         frames: Sequence[str]) -> float:
    """Probability a worse-performing frame scores higher, ties counted half.

    Computed over every pair rather than by ranking, because the frame counts
    here are small and the pairwise form is the definition rather than a
    shortcut to it.
    """
    better = total = 0.0
    for i, a in enumerate(frames):
        for b in frames[i + 1:]:
            if performance[a] == performance[b]:
                continue
            worse, healthier = ((a, b) if performance[a] < performance[b]
                                else (b, a))
            total += 1.0
            if scores[worse] > scores[healthier]:
                better += 1.0
            elif scores[worse] == scores[healthier]:
                better += 0.5
    return better / total if total else 0.5


def triggering_candidates(scores: dict[str, float], threshold: float
                          ) -> tuple[str, ...]:
    """Frames outside the envelope, furthest first.

    Named "candidates" rather than "triggering conditions" on purpose. A
    triggering condition in ISO 21448 is a scenario, described and reasoned
    about by a person. This returns frames that warrant that look. Promoting
    the output of a statistic straight into a safety artefact is the shortcut
    this name refuses to take.
    """
    return tuple(frame for frame, _ in
                 sorted(((f, s) for f, s in scores.items() if s > threshold),
                        key=lambda pair: -pair[1]))
