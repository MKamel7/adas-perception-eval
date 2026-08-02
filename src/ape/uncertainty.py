"""How much of a slice difference is real, and how much is the sample.

THE PROBLEM THIS EXISTS TO FIX. The report puts a number computed from 12
objects next to one computed from 5579 and distinguishes them with a tilde.
That is not enough. "Pedestrians beyond 50 m score 0.000" and "pedestrians
within 10 m score 0.689" is a strong claim; whether it survives contact with
sampling noise is a question the point estimate cannot answer, and a validation
engineer who cannot answer it should not be quoting the number.

So every AP is reported with a confidence interval obtained by bootstrapping.

RESAMPLING FRAMES, NOT OBJECTS, and this is the part that is easy to get wrong.
Objects within one frame are not independent: the same lighting, the same
occluders, the same camera pose, often the same pedestrian group. Resampling
objects would treat twelve people standing together as twelve independent
observations, which understates the interval, sometimes badly. The frame is the
unit that was sampled from the world, so the frame is the unit that gets
resampled.

WHY THERE IS A SECOND AP IMPLEMENTATION IN HERE, which would otherwise be
exactly the duplication this project argues against. Bootstrapping needs
hundreds of recomputations per slice and the reference implementation in
`metrics.py` is written for clarity. This one is vectorised. It is NOT trusted
on that basis: `tests/test_uncertainty.py` requires it to agree with
`metrics.average_precision` exactly, on random inputs, so the fast path is
checked against the path that was itself checked against pycocotools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ape.match import Assignment
from ape.metrics import RECALL_POINTS

#: Enough resamples for a stable 95% interval without making the report slow.
#: 1000 moves the bounds by well under 0.005 against 200 here, which is finer
#: than the numbers are quoted to.
RESAMPLES = 400

#: The interval reported. 95% is convention rather than a considered choice, and
#: is stated so nobody mistakes it for one.
ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range the resampling put around it."""

    point: float
    low: float
    high: float
    #: Frames that contributed. The bootstrap resamples these, so an interval
    #: computed from three frames is wide for a reason and should say so.
    frames: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def separated_from(self, other: Interval) -> bool:
        """Do the two intervals fail to overlap?

        The honest weak form of "is this difference real". Non-overlapping
        intervals are sufficient evidence of a difference; overlapping ones are
        NOT evidence of no difference, and the report says so rather than
        letting an overlap be read as a null result.
        """
        return self.high < other.low or other.high < self.low


def _ap(scores: np.ndarray, hits: np.ndarray, positives: int) -> float:
    """COCO 101-point AP, vectorised. Must match `metrics.average_precision`."""
    if positives == 0:
        return float("nan")
    if scores.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    ordered = hits[order]
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, ordered.size + 1)
    recall = cumulative / positives

    # Monotone envelope from the right, the same operation the reference does
    # with a reversed loop.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]

    # For each recall point, the precision at the first index reaching it.
    positions = np.searchsorted(recall, np.asarray(RECALL_POINTS), side="left")
    inside = positions < envelope.size
    interpolated = np.zeros(len(RECALL_POINTS))
    interpolated[inside] = envelope[positions[inside]]
    return float(interpolated.mean())


def bootstrap(per_frame: dict[str, Assignment],
              resamples: int = RESAMPLES,
              alpha: float = ALPHA,
              seed: int = 20260802) -> Interval:
    """Resample frames with replacement and report the interval on AP.

    Seeded, because an interval that moves between runs of the same evaluation
    would be indistinguishable from one that moved because the detector changed.
    """
    frames = list(per_frame)
    if not frames:
        return Interval(float("nan"), float("nan"), float("nan"), 0)

    scores = [np.array([s for s, _ in per_frame[f].scored], dtype=np.float64)
              for f in frames]
    hits = [np.array([t for _, t in per_frame[f].scored], dtype=np.float64)
            for f in frames]
    positives = np.array([per_frame[f].positives for f in frames])

    point = _ap(np.concatenate(scores) if scores else np.array([]),
                np.concatenate(hits) if hits else np.array([]),
                int(positives.sum()))

    rng = np.random.default_rng(seed)
    drawn = np.empty(resamples)
    count = len(frames)
    for i in range(resamples):
        pick = rng.integers(0, count, count)
        total = int(positives[pick].sum())
        if total == 0:
            drawn[i] = np.nan
            continue
        drawn[i] = _ap(np.concatenate([scores[j] for j in pick]),
                       np.concatenate([hits[j] for j in pick]), total)

    usable = drawn[~np.isnan(drawn)]
    if usable.size == 0:
        return Interval(point, float("nan"), float("nan"), count)
    low, high = np.quantile(usable, [alpha / 2, 1 - alpha / 2])
    return Interval(point, float(low), float(high), count)
