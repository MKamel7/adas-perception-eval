"""Is the detector's confidence worth believing?

WHY THIS IS A SEPARATE QUESTION FROM ACCURACY. mAP asks how often the detector
is right. Calibration asks whether it KNOWS how often it is right. A detector at
0.68 mAP that reports 0.95 on every box it will get wrong is a different and
worse engineering problem than one at 0.68 that reports 0.4 on those boxes,
because the second can be gated by a threshold and the first cannot. Nothing
else in this repository measures that.

WHY SOTIF CARES ABOUT THE SIGN, and why plain ECE is not enough on its own.
Expected calibration error is the mean gap between confidence and observed
precision, and it is SYMMETRIC: a detector that says 0.6 and is right 0.9 of the
time scores exactly as badly as one that says 0.9 and is right 0.6 of the time.
Those are not equally dangerous. The first is timid and a downstream planner
merely wastes performance on it; the second is CONFIDENTLY WRONG, which is the
failure ISO 21448 exists for. So `overconfidence_error` reports only the gaps in
the dangerous direction, and it is the number to read first.

WHAT A "TRUE POSITIVE" MEANS HERE is whatever `ape.match` already decided, at
the IoU threshold it was given. Calibration is measured against the SAME verdict
the metrics use, rather than against a second opinion computed here, because two
definitions of correctness in one repository is how a calibration figure ends up
disagreeing with the precision figure it is supposed to explain.

IGNORED DETECTIONS ARE EXCLUDED. KITTI's neutral classes are neither credited
nor penalised by the matcher, so counting them as failures would manufacture
overconfidence out of a labelling convention.

SLICING IS BY THE DETECTION, NOT BY THE GROUND TRUTH, which is the one design
decision here worth arguing with. Every other slice in this repository is cut on
ground-truth attributes: range, occlusion, truncation. Those exist only for
objects that are really there, so slicing calibration that way would silently
drop every false positive. False positives are exactly where overconfidence
does its damage, so the bands here are cut on properties the detection itself
has, and box height is used as the range proxy. It is a weaker proxy than a
labelled distance and it is the only one available for a box that corresponds to
nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

from ape.match import Judged

#: Ten equal-width bins over [0, 1]. Ten is the common default in the
#: calibration literature and the choice matters: too few hides a bad region by
#: averaging it away, too many leaves bins with two detections in them whose
#: observed precision is 0.0 or 1.0 and nothing in between. `bins=` is exposed
#: so a reader can see the answer move.
DEFAULT_BINS = 10


@dataclass(frozen=True)
class Bin:
    """One confidence band, and what actually happened inside it."""

    lower: float
    upper: float
    count: int
    mean_confidence: float
    precision: float

    @property
    def gap(self) -> float:
        """Signed. Positive means the detector claimed more than it delivered."""
        return self.mean_confidence - self.precision

    @property
    def overconfident(self) -> bool:
        return self.gap > 0.0


@dataclass(frozen=True)
class Reliability:
    """A reliability diagram as data, plus the scalars that summarise it.

    The bins are the diagram: plotting `mean_confidence` against `precision`
    gives the usual picture, and the diagonal is perfect calibration. They are
    kept rather than discarded so a reader can see WHERE the detector is wrong
    rather than only how wrong on average, which is the same argument this
    repository makes about mAP.
    """

    bins: tuple[Bin, ...]
    total: int

    @property
    def populated(self) -> tuple[Bin, ...]:
        """Bins with something in them. An empty bin has no observed precision."""
        return tuple(b for b in self.bins if b.count)

    @property
    def expected_calibration_error(self) -> float:
        """Count-weighted mean absolute gap. Symmetric, and that is its limit."""
        if not self.total:
            return 0.0
        return sum(b.count * abs(b.gap) for b in self.populated) / self.total

    @property
    def overconfidence_error(self) -> float:
        """Count-weighted mean gap, counting only the dangerous direction.

        The number to read first. A detector can post a respectable ECE while
        every one of its errors is overconfidence, and this separates the two.
        """
        if not self.total:
            return 0.0
        return sum(b.count * max(b.gap, 0.0)
                   for b in self.populated) / self.total

    @property
    def maximum_calibration_error(self) -> float:
        """The worst single bin, unweighted.

        Worth reporting next to ECE because a small, badly calibrated,
        high-confidence bin is precisely the region a threshold will select.
        """
        return max((abs(b.gap) for b in self.populated), default=0.0)

    @property
    def worst_bin(self) -> Bin | None:
        """The bin driving `maximum_calibration_error`, or None if empty."""
        populated = self.populated
        if not populated:
            return None
        return max(populated, key=lambda b: abs(b.gap))


def scored(judged: Iterable[Judged]) -> list[tuple[float, bool]]:
    """(confidence, was it right) for every detection that counts.

    Ignored detections are dropped: the matcher neither credits nor penalises
    them, so calling them wrong would manufacture overconfidence out of a
    labelling convention.
    """
    return [(j.detection.score, j.true_positive) for j in judged if not j.ignored]


def reliability(judged: Iterable[Judged], bins: int = DEFAULT_BINS
                ) -> Reliability:
    """Bin detections by confidence and measure what each band delivered."""
    if bins < 1:
        raise ValueError(f"bins must be at least 1, got {bins}")

    observations = scored(judged)
    edges = [i / bins for i in range(bins + 1)]
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]

    for score, hit in observations:
        # Clamped so a score of exactly 1.0 lands in the top bin rather than in
        # a bin that does not exist, and a score outside [0, 1] from some future
        # detector does not silently vanish.
        index = min(bins - 1, max(0, int(score * bins)))
        buckets[index].append((score, hit))

    out: list[Bin] = []
    for index, bucket in enumerate(buckets):
        count = len(bucket)
        mean = sum(s for s, _ in bucket) / count if count else 0.0
        hits = sum(1 for _, h in bucket if h)
        out.append(Bin(lower=edges[index], upper=edges[index + 1], count=count,
                       mean_confidence=mean,
                       precision=hits / count if count else 0.0))
    return Reliability(bins=tuple(out), total=len(observations))


def by_slice(judged: Iterable[Judged],
             binner: Callable[[Judged], str | None],
             bins: int = DEFAULT_BINS) -> dict[str, Reliability]:
    """Calibration per band, using a binner over the DETECTIONS.

    `binner` returning None drops a detection from every slice, which is how a
    caller says "this one does not belong to any band" without inventing an
    "other" bucket that then gets read as a real population.
    """
    grouped: dict[str, list[Judged]] = {}
    for item in judged:
        band = binner(item)
        if band is not None:
            grouped.setdefault(band, []).append(item)
    return {band: reliability(items, bins)
            for band, items in sorted(grouped.items())}


#: Box height in pixels is the range proxy. Bigger box, nearer object. It is
#: weaker than KITTI's labelled distance and it is the only thing available for
#: a false positive, which corresponds to no object and therefore has no range.
HEIGHT_EDGES: tuple[float, ...] = (25.0, 40.0, 80.0)


def height_band(item: Judged) -> str:
    """Which height band a detection's own box falls in."""
    height = item.detection.box.height
    edges = HEIGHT_EDGES
    if height < edges[0]:
        return f"under {edges[0]:.0f}px"
    for low, high in pairwise(edges):
        if height < high:
            return f"{low:.0f} to {high:.0f}px"
    return f"over {edges[-1]:.0f}px"


def label_band(item: Judged) -> str:
    return item.detection.label


def overconfident_slices(slices: dict[str, Reliability],
                         threshold: float) -> tuple[str, ...]:
    """Bands whose overconfidence exceeds a budget, worst first.

    Returned as names rather than as a pass/fail, because the useful output of
    a calibration check is WHICH band to distrust. A single boolean over the
    whole set is the aggregate this repository exists to argue against.
    """
    over = [(name, r.overconfidence_error) for name, r in slices.items()
            if r.overconfidence_error > threshold]
    return tuple(name for name, _ in sorted(over, key=lambda p: -p[1]))


def reliability_table(result: Reliability) -> Sequence[tuple[str, ...]]:
    """The diagram as rows, for a report or a docstring.

    Empty bins are kept and marked. Dropping them would let a detector that
    never emits a confidence between 0.3 and 0.7 look like one whose behaviour
    there was measured and fine.
    """
    rows: list[tuple[str, ...]] = [
        ("band", "n", "confidence", "precision", "gap")]
    for b in result.bins:
        if not b.count:
            rows.append((f"{b.lower:.1f} to {b.upper:.1f}", "0", "", "", ""))
            continue
        rows.append((f"{b.lower:.1f} to {b.upper:.1f}", str(b.count),
                     f"{b.mean_confidence:.3f}", f"{b.precision:.3f}",
                     f"{b.gap:+.3f}"))
    return rows
