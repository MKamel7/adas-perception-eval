"""Assign detections to ground truth, and decide what each one was.

Three outcomes, not two, and the third is the one people get wrong.

  true positive   matched an object of the class being evaluated
  false positive  matched nothing it was allowed to match
  IGNORED         matched something the benchmark declined to score

Ignoring is not a convenience. KITTI publishes the rules: a detection landing on
a Van must not count against a Car detector, one landing on a DontCare region has
not made a mistake, and an object too small or too occluded for the difficulty
tier being reported is not a miss. Treating any of those as a false positive
inflates the error for every detector equally, which is worse than useless
because the result still looks real.

MATCHING ORDER IS PART OF THE DEFINITION. Detections are taken in descending
score and each takes the best available ground truth. Match greedily in file
order instead and a low-confidence box can steal the object a confident one
would have claimed, which lowers AP for reasons that have nothing to do with the
detector. This mirrors what the reference COCO implementation does, deliberately,
because `tests/test_metrics_against_reference.py` compares the two and a
difference in matching would show up there as a difference in the metric and be
much harder to locate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ape.records import Detection, Difficulty, GroundTruth

#: Does this ground truth object count towards the subset being reported?
type Predicate = Callable[[GroundTruth], bool]

#: The tiers a difficulty is scored under. KITTI's tiers are cumulative: an
#: object that is Easy is also counted when reporting Moderate and Hard, so
#: "Hard" means "everything up to and including hard", not "hard only".
INCLUDED_AT: dict[Difficulty, frozenset[Difficulty]] = {
    Difficulty.EASY: frozenset({Difficulty.EASY}),
    Difficulty.MODERATE: frozenset({Difficulty.EASY, Difficulty.MODERATE}),
    Difficulty.HARD: frozenset({Difficulty.EASY, Difficulty.MODERATE,
                                Difficulty.HARD}),
}


@dataclass
class Assignment:
    """What happened to every detection of one class, in one evaluation."""

    #: (score, is_true_positive) for every detection that was scored at all.
    #: Ignored detections are absent: they are neither right nor wrong.
    scored: list[tuple[float, bool]] = field(default_factory=list)
    #: Ground truth objects that COULD have been found. The denominator of
    #: recall, and the single easiest number to get quietly wrong.
    positives: int = 0
    #: Detections dropped for landing on something unscored. Reported so the
    #: count can be sanity-checked rather than taken on trust.
    ignored: int = 0

    def extend(self, other: Assignment) -> None:
        self.scored.extend(other.scored)
        self.positives += other.positives
        self.ignored += other.ignored


def at_difficulty(tier: Difficulty) -> Predicate:
    """Objects this tier claims to find. Cumulative, as KITTI defines it."""
    allowed = INCLUDED_AT[tier]
    return lambda item: item.difficulty in allowed


def in_slice(binner: Callable[[GroundTruth], str | None], value: str) -> Predicate:
    """Objects falling in one bin of one slice dimension."""
    return lambda item: binner(item) == value


def partition(ground_truth: list[GroundTruth], label: str,
              neutral: frozenset[str],
              counts_when: Predicate | None = None
              ) -> tuple[list[GroundTruth], list[GroundTruth]]:
    """Split ground truth into what counts and what is merely tolerated.

    An object of the right class that falls OUTSIDE the subset being reported,
    a harder tier, another distance band, lands in the tolerated pile rather
    than being dropped. That distinction is the whole point of slicing:

      dropping it     the object vanishes, and a detection that found it
                      becomes a false positive for being correct

      counting it     the slice is punished for missing objects it never
                      claimed to be about

      tolerating it   neither, which is the only honest answer

    This is why every slice below reuses this function instead of filtering the
    ground truth list, which is the obvious approach and is wrong in both
    directions at once.
    """
    counts: list[GroundTruth] = []
    tolerated: list[GroundTruth] = []

    for item in ground_truth:
        if item.label == label:
            if counts_when is None or counts_when(item):
                counts.append(item)
            else:
                tolerated.append(item)
        elif item.label in neutral:
            tolerated.append(item)
    return counts, tolerated


def assign_frame(detections: list[Detection], ground_truth: list[GroundTruth],
                 label: str, neutral: frozenset[str], iou_threshold: float,
                 counts_when: Predicate | None = None) -> Assignment:
    """One frame, one class."""
    counts, tolerated = partition(ground_truth, label, neutral, counts_when)
    result = Assignment(positives=len(counts))

    claimed: set[int] = set()
    for detection in sorted(detections, key=lambda d: d.score, reverse=True):
        if detection.label != label:
            continue

        best_iou, best_index = 0.0, -1
        for index, candidate in enumerate(counts):
            if index in claimed:
                continue
            overlap = detection.box.iou(candidate.box)
            if overlap > best_iou:
                best_iou, best_index = overlap, index

        if best_index >= 0 and best_iou >= iou_threshold:
            claimed.add(best_index)
            result.scored.append((detection.score, True))
            continue

        # Nothing scoreable. Before calling it a mistake, check whether it
        # landed on something the benchmark refuses to score.
        if any(detection.box.iou(item.box) >= iou_threshold for item in tolerated):
            result.ignored += 1
            continue

        result.scored.append((detection.score, False))

    return result


def assign(detections: dict[str, list[Detection]],
           ground_truth: dict[str, list[GroundTruth]],
           label: str, neutral: frozenset[str], iou_threshold: float,
           counts_when: Predicate | None = None) -> Assignment:
    """Every frame, one class.

    Frames present in the ground truth but absent from the detections are still
    evaluated, with no detections. Skipping them would silently drop their
    objects from the recall denominator and report a detector that found
    everything it looked at as one that found everything.
    """
    total = Assignment()
    for frame_id, truths in ground_truth.items():
        total.extend(assign_frame(detections.get(frame_id, []), truths,
                                  label, neutral, iou_threshold, counts_when))
    return total
