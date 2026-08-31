"""What kind of mistake was it? The taxonomy for the false positives.

THE HALF OF THE ERROR STORY THIS PROJECT WAS NOT TELLING. `ape.localisation`
already splits the MISSES into "never seen" and "seen and boxed badly", and
says why that matters: they have different fixes. The false positives had no
such split at all. Every wrong box counted the same, so a detector that fires
twice on one pedestrian and a detector that invents pedestrians in empty road
produced the same number, and AP alone cannot tell them apart either.

They are not the same problem:

  duplicate       the object IS there and was already found by a better-scoring
                  box. The detector is right about the world and wrong about
                  how many things are in it. Fixes with non-maximum suppression
                  and costs almost nothing in safety terms.

  misclassified   right place, wrong label: a box on a real object of another
                  scored class. The detector saw something and called it the
                  wrong thing, which for a vehicle is the difference between
                  braking for a pedestrian and braking for a car, and both are
                  braking.

  mislocalised    right class, real object underneath, box too loose to count.
                  Fixes with box regression, and the system still knows
                  something is there.

  hallucinated    nothing was there. THE ONE THAT MATTERS MOST for a safety
                  argument, because it is the only category that makes a
                  vehicle brake for empty road, and it is the only one whose
                  cause is not visible anywhere in the ground truth.

An AP of 0.6 made mostly of duplicates and an AP of 0.6 made mostly of
hallucinations describe different systems. Reporting one number for both is the
gap this module closes.

WHY IT DOES NOT MATCH ANYTHING ITSELF. Every verdict here is read off the
`Judged` records `ape.match.judge_frame` already produced, at the same
threshold, in the same order. Re-matching would be a second matcher and a
second matcher drifts, which is exactly the fault that made this refactor
necessary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from ape.match import FrameOutcome, Judged, judge_frame
from ape.records import Detection, GroundTruth

#: Below this a box is not "on" anything in a useful sense, and calling it a
#: mislocalisation would credit the detector for a coincidental overlap. Same
#: value and same reasoning as `ape.localisation.TOUCHING`.
TOUCHING = 0.1


class Outcome(StrEnum):
    """What one detection turned out to be."""

    TRUE_POSITIVE = "true positive"
    DUPLICATE = "duplicate"
    MISCLASSIFIED = "misclassified"
    MISLOCALISED = "mislocalised"
    HALLUCINATED = "hallucinated"
    #: Landed on something the benchmark declines to score. Not a mistake.
    IGNORED = "ignored"


#: The failure kinds, worst last. Order is the reporting order and is chosen so
#: the category a safety case cares about most sits at the end of the row.
FAILURES = (Outcome.DUPLICATE, Outcome.MISCLASSIFIED,
            Outcome.MISLOCALISED, Outcome.HALLUCINATED)


@dataclass(frozen=True)
class Verdict:
    """One detection and what it turned out to be."""

    detection: Detection
    outcome: Outcome
    #: The overlap that decided it, for reading a borderline case by hand.
    iou: float


@dataclass
class Breakdown:
    """Every detection of one class, sorted into the taxonomy."""

    label: str
    iou_threshold: float
    counts: Counter[Outcome] = field(default_factory=Counter)

    @property
    def scored(self) -> int:
        """Detections that counted either way. Ignored ones are neither."""
        return sum(n for outcome, n in self.counts.items()
                   if outcome is not Outcome.IGNORED)

    @property
    def false_positives(self) -> int:
        return sum(self.counts[outcome] for outcome in FAILURES)

    def share(self, outcome: Outcome) -> float:
        """This outcome as a fraction of all false positives.

        The fraction, not the count, is what compares two detectors: a model
        with twice the boxes has twice of everything.
        """
        total = self.false_positives
        return self.counts[outcome] / total if total else float("nan")

    def extend(self, other: Breakdown) -> None:
        self.counts.update(other.counts)


def _verdict(judged: Judged, outcome: FrameOutcome,
             others: list[GroundTruth], iou_threshold: float) -> Verdict:
    if judged.ignored:
        return Verdict(judged.detection, Outcome.IGNORED, judged.best_free)
    if judged.true_positive:
        return Verdict(judged.detection, Outcome.TRUE_POSITIVE, judged.best_free)

    # Already claimed by a better-scoring box: the object is real and was
    # found. best_any sees claimed ground truth, best_free does not.
    if judged.best_any >= iou_threshold:
        return Verdict(judged.detection, Outcome.DUPLICATE, judged.best_any)

    # Right place, wrong label. Checked before mislocalisation because a box
    # sitting squarely on a car is a naming mistake, not a regression problem,
    # even when it also clips a pedestrian.
    on_other = max((judged.detection.box.iou(item.box) for item in others),
                   default=0.0)
    if on_other >= iou_threshold:
        return Verdict(judged.detection, Outcome.MISCLASSIFIED, on_other)

    # Right class, real object underneath, box too loose to count.
    if judged.best_free >= TOUCHING:
        return Verdict(judged.detection, Outcome.MISLOCALISED, judged.best_free)

    return Verdict(judged.detection, Outcome.HALLUCINATED, judged.best_free)


def classify_frame(detections: list[Detection], ground_truth: list[GroundTruth],
                   label: str, neutral: frozenset[str], iou_threshold: float,
                   scored_labels: tuple[str, ...] = ()) -> list[Verdict]:
    """Sort one frame's detections of one class into the taxonomy.

    `scored_labels` is the full set of classes the project evaluates, needed
    because misclassification is the one verdict that cannot be reached from
    this class's own ground truth: deciding a box is a mislabelled car means
    looking at the cars.
    """
    outcome = judge_frame(detections, ground_truth, label, neutral, iou_threshold)
    others = [item for item in ground_truth
              if item.label != label and item.label in scored_labels]
    return [_verdict(judged, outcome, others, iou_threshold)
            for judged in outcome.judged]


def classify(detections: dict[str, list[Detection]],
             ground_truth: dict[str, list[GroundTruth]],
             label: str, neutral: frozenset[str], iou_threshold: float,
             scored_labels: tuple[str, ...] = ()) -> Breakdown:
    """Every frame, one class.

    Frames in the ground truth with no detections are still visited, matching
    `ape.match.assign`, so the two agree on which frames exist.
    """
    result = Breakdown(label=label, iou_threshold=iou_threshold)
    for frame_id, truths in ground_truth.items():
        for verdict in classify_frame(detections.get(frame_id, []), truths,
                                      label, neutral, iou_threshold,
                                      scored_labels):
            result.counts[verdict.outcome] += 1
    return result
