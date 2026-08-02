"""Average precision, computed here and checked against the reference.

THIS FILE IS THE ONE THAT HAS TO BE RIGHT, and the only way to know it is right
is to compare it with an implementation nobody here wrote.
`tests/test_metrics_against_reference.py` runs the same detections through
`pycocotools` and requires agreement to within 0.001. That comparison is the
deliverable; this code is just one side of it.

`pycocotools` is a dependency of the TESTS and never of this module. If the
measurement path imported the reference, the comparison would be circular and
would prove nothing.

WHY 101-POINT INTERPOLATION rather than the older 11-point rule or the exact
area under the curve. It is what COCO defines, so it is what the reference
computes, so it is the only choice that makes agreement to 0.001 a meaningful
statement rather than a coincidence of two arbitrary conventions. KITTI's own
tooling uses a 40-point variant; that is noted in the report rather than mixed
in here, because averaging two conventions produces a number that matches
neither.
"""

from __future__ import annotations

from dataclasses import dataclass

from ape.match import Assignment

#: COCO's recall grid: 0.00, 0.01, ... 1.00.
RECALL_POINTS = tuple(i / 100 for i in range(101))


@dataclass(frozen=True)
class Curve:
    """A precision-recall curve, and the single number summarising it."""

    precision: tuple[float, ...]
    recall: tuple[float, ...]
    average_precision: float
    positives: int
    true_positives: int
    false_positives: int

    @property
    def best_recall(self) -> float:
        return self.recall[-1] if self.recall else 0.0


def precision_recall(assignment: Assignment) -> tuple[list[float], list[float]]:
    """Running precision and recall, in descending score order.

    Ties in score are NOT reordered. Two detections with identical confidence
    have no defined order, and imposing one would make the curve depend on
    dictionary iteration order, which is exactly the kind of instability that
    makes a metric disagree with itself between runs.
    """
    ordered = sorted(assignment.scored, key=lambda item: item[0], reverse=True)

    precision: list[float] = []
    recall: list[float] = []
    hits = 0
    for index, (_, is_true_positive) in enumerate(ordered, 1):
        hits += is_true_positive
        precision.append(hits / index)
        recall.append(hits / assignment.positives if assignment.positives else 0.0)
    return precision, recall


def average_precision(assignment: Assignment) -> Curve:
    """COCO-style AP: the mean of interpolated precision over 101 recalls."""
    precision, recall = precision_recall(assignment)
    hits = sum(1 for _, tp in assignment.scored if tp)
    misses = len(assignment.scored) - hits

    if not assignment.positives:
        # No objects to find. AP is undefined rather than zero, and returning
        # zero would drag any average containing this class downwards for a
        # class that was never present.
        return Curve((), (), float("nan"), 0, hits, misses)

    if not precision:
        return Curve((), (), 0.0, assignment.positives, 0, 0)

    # Make precision monotonically non-increasing from the right. Without this
    # the curve is jagged and the interpolated value at a recall point depends
    # on a single detection rather than on the best precision achievable at
    # that recall or beyond, which is what AP is defined over.
    envelope = list(precision)
    for i in range(len(envelope) - 2, -1, -1):
        envelope[i] = max(envelope[i], envelope[i + 1])

    interpolated: list[float] = []
    position = 0
    for point in RECALL_POINTS:
        while position < len(recall) and recall[position] < point:
            position += 1
        interpolated.append(envelope[position] if position < len(envelope) else 0.0)

    return Curve(tuple(precision), tuple(recall),
                 sum(interpolated) / len(interpolated),
                 assignment.positives, hits, misses)


def mean_average_precision(curves: dict[str, Curve]) -> float:
    """Mean AP over classes, skipping classes that were never present.

    A class with no ground truth has an undefined AP, not a zero. Averaging a
    zero in would report a detector as worse for being evaluated on a split that
    happens to contain no trams.
    """
    defined = [c.average_precision for c in curves.values()
               if c.positives and c.average_precision == c.average_precision]
    return sum(defined) / len(defined) if defined else float("nan")
