"""Run the whole evaluation: overall, per difficulty tier, and per slice.

The aggregate number is computed first and then largely ignored, which is the
argument this project is making. A detector reported at one number can be
near-blind on a subset that matters, and the only way to know is to cut the same
data along attributes the benchmark annotated before anyone saw a result.

Every figure here comes out of the same `assign` and `average_precision` that
`tests/test_metrics_against_reference.py` checks against pycocotools. There is
no second implementation for slices, because a slice computed by different code
than the aggregate would make any difference between them uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ape.classes import EVALUATED, HEADLINE, neutral_labels
from ape.localisation import THRESHOLDS, Diagnosis, diagnose
from ape.match import Assignment, assign, assign_by_frame, at_difficulty, in_slice
from ape.metrics import Curve, average_precision, mean_average_precision
from ape.operating import Point, best_recall, sweep, threshold_for_recall
from ape.outcomes import Breakdown, classify
from ape.records import Detection, Difficulty, GroundTruth
from ape.slices import DIMENSIONS
from ape.uncertainty import Interval, bootstrap

#: The operating point everything is reported at. KITTI's 2D benchmark uses 0.7
#: for Car and 0.5 for pedestrians and cyclists; a single threshold is used here
#: so that classes are comparable with each other, and the difference from
#: KITTI's convention is stated in the report rather than left to be discovered.
IOU = 0.5


@dataclass(frozen=True)
class Cell:
    """One number, and enough context to know whether to believe it."""

    label: str
    curve: Curve
    #: The 95% bootstrap interval, or None when it was not computed.
    #:
    #: This is what replaced counting objects as the measure of confidence. A
    #: floor of ten objects was a proxy for "is this number stable", and a
    #: crude one: ten objects in ten frames and ten objects in one frame are
    #: not equally informative, and the object count cannot tell them apart.
    interval: Interval | None = None

    @property
    def trustworthy(self) -> bool:
        """Enough objects for the number to mean anything.

        Kept as a coarse floor for the cells where no interval was computed.
        Where an interval exists, its WIDTH is the honest answer and this is
        only a fallback.
        """
        return self.curve.positives >= 10


@dataclass(frozen=True)
class SliceResult:
    dimension: str
    question: str
    bin: str
    cells: tuple[Cell, ...]

    def cell(self, label: str) -> Cell | None:
        return next((c for c in self.cells if c.label == label), None)


@dataclass
class Evaluation:
    frames: int = 0
    objects: int = 0
    overall: dict[str, Curve] = field(default_factory=dict)
    overall_interval: dict[str, Interval] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Curve]] = field(default_factory=dict)
    #: The same 95% bootstrap interval the overall figures and the slice cells
    #: carry. The difficulty tiers are a slice like any other, and reporting
    #: them as bare numbers beside slices that show a range invited exactly the
    #: comparison the intervals exist to prevent: Easy against Hard looks like a
    #: finding until you see how much of the gap the sample explains.
    by_difficulty_interval: dict[str, dict[str, Interval]] = field(
        default_factory=dict)
    slices: list[SliceResult] = field(default_factory=list)
    #: class -> the operating-point curve. AP integrates over every threshold;
    #: a vehicle runs at one, and this is where that choice becomes visible.
    curve_points: dict[str, list[Point]] = field(default_factory=dict)
    #: class -> the most recall reachable at ANY threshold. Below a target,
    #: no threshold choice helps and the answer is a different sensor.
    ceiling: dict[str, float] = field(default_factory=dict)
    #: iou threshold -> class -> AP. 0.7 is KITTI's own for Car, so this is
    #: what makes the numbers comparable with the KITTI benchmark instead of
    #: only with each other.
    at_iou: dict[float, dict[str, float]] = field(default_factory=dict)
    #: class -> why the objects missed at IoU 0.5 were missed. Not seen at all,
    #: or seen and boxed badly: different fixes, different severities.
    diagnosis: dict[str, Diagnosis] = field(default_factory=dict)
    #: class -> what KIND of mistake each false positive was. The other half of
    #: the same question: `diagnosis` explains the misses, this explains the
    #: wrong boxes. An AP made of duplicates and an AP made of hallucinations
    #: describe different systems and one number cannot tell them apart.
    outcomes: dict[str, Breakdown] = field(default_factory=dict)

    @property
    def headline(self) -> float:
        return mean_average_precision(
            {k: v for k, v in self.overall.items() if k in HEADLINE})

    def spread(self, label: str) -> tuple[float, float, str, str]:
        """Best and worst trustworthy slice for a class, and where they were.

        The headline finding of the whole project is this ratio, not the
        aggregate, so it is computed rather than eyeballed off a table.
        """
        found = [(s, c) for s in self.slices
                 if (c := s.cell(label)) and c.trustworthy
                 and c.curve.average_precision == c.curve.average_precision]
        if not found:
            return (float("nan"), float("nan"), "", "")
        best = max(found, key=lambda pair: pair[1].curve.average_precision)
        worst = min(found, key=lambda pair: pair[1].curve.average_precision)
        return (best[1].curve.average_precision, worst[1].curve.average_precision,
                f"{best[0].dimension}: {best[0].bin}",
                f"{worst[0].dimension}: {worst[0].bin}")


#: The recall targets the report asks about. Not recommendations: what a
#: function needs depends on the vehicle, the speed and the regulator, none of
#: which are in this repository. They are the questions, and the table is the
#: answer to each.
RECALL_TARGETS = (0.50, 0.70, 0.80, 0.90, 0.95)


def operating_table(result: Evaluation, label: str
                    ) -> list[tuple[float, Point | None]]:
    """What each target recall would cost, or that it cannot be bought."""
    points = result.curve_points.get(label, [])
    return [(target, threshold_for_recall(points, target))
            for target in RECALL_TARGETS]


def evaluate(detections: dict[str, list[Detection]],
             truth: dict[str, list[GroundTruth]],
             iou: float = IOU) -> Evaluation:
    result = Evaluation(frames=len(truth),
                        objects=sum(len(v) for v in truth.values()))

    for label in EVALUATED:
        neutral = neutral_labels(label)
        result.overall[label] = average_precision(
            assign(detections, truth, label, neutral, iou))
        result.overall_interval[label] = bootstrap(
            assign_by_frame(detections, truth, label, neutral, iou))

        for threshold in THRESHOLDS:
            result.at_iou.setdefault(threshold, {})[label] = average_precision(
                assign(detections, truth, label, neutral, threshold)
            ).average_precision
        result.diagnosis[label] = diagnose(detections, truth, label, neutral,
                                           tight=iou)
        result.outcomes[label] = classify(detections, truth, label, neutral,
                                          iou, EVALUATED)

        whole = assign(detections, truth, label, neutral, iou)
        result.curve_points[label] = sweep(whole, len(truth))
        result.ceiling[label] = best_recall(result.curve_points[label])

        for tier in (Difficulty.EASY, Difficulty.MODERATE, Difficulty.HARD):
            result.by_difficulty.setdefault(tier.value, {})[label] = \
                average_precision(assign(detections, truth, label, neutral, iou,
                                         at_difficulty(tier)))

    for dimension in DIMENSIONS:
        for name in dimension.bins:
            cells = []
            for label in EVALUATED:
                predicate = in_slice(dimension.of, str(name))
                per_frame = assign_by_frame(detections, truth, label,
                                            neutral_labels(label), iou, predicate)
                assignment = Assignment()
                for piece in per_frame.values():
                    assignment.extend(piece)
                # Only where there is something to be uncertain about. A slice
                # with no objects has no AP, so it has no interval either.
                interval = (bootstrap({k: v for k, v in per_frame.items()
                                       if v.positives or v.scored})
                            if assignment.positives else None)
                cells.append(Cell(label, average_precision(assignment), interval))
            result.slices.append(SliceResult(dimension.name, dimension.question,
                                             str(name), tuple(cells)))
    return result
