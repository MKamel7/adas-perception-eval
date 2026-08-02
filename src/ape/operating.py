"""Where would you actually set the threshold, and what does it cost?

AVERAGE PRECISION CANNOT ANSWER THIS, and that is not a criticism of it. AP
integrates over every confidence threshold at once, which is exactly what you
want for comparing two detectors and exactly what you cannot ship. A vehicle
runs at ONE threshold. Choosing it is the decision that turns an evaluation into
an engineering argument, and this module is the part of the project that makes
that decision visible.

THE TRADE, in safety terms rather than machine-learning ones:

  threshold too high   pedestrians go unreported. The hazard is a collision,
                       and the cost is measured in missed objects.

  threshold too low    the vehicle reacts to things that are not there. The
                       hazard is an unnecessary intervention, phantom braking
                       on a motorway, and the cost is measured in false
                       detections per frame.

Both are hazards. A report that only counted misses would be arguing for a
threshold of zero, which would brake continuously. So both directions are
reported, and the false alarm rate is given PER FRAME rather than as precision,
because "one phantom brake every fourteen frames" is a quantity an integrator
can reason about and "precision 0.82" is not.

WHAT THIS DELIBERATELY DOES NOT DO: recommend a threshold. That depends on the
vehicle, the speed, the function and the regulatory context, none of which are
in this repository. It reports the curve and the cost of each point on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from ape.match import Assignment


@dataclass(frozen=True)
class Point:
    """One operating point: what you catch, and what it costs you."""

    threshold: float
    recall: float
    precision: float
    true_positives: int
    false_positives: int
    missed: int
    frames: int

    @property
    def false_alarms_per_frame(self) -> float:
        """The number an integrator can act on.

        Precision is a ratio between two things the detector did. This is a
        rate against the world: how often the vehicle would react to nothing.
        """
        return self.false_positives / self.frames if self.frames else float("nan")

    @property
    def frames_per_false_alarm(self) -> float:
        rate = self.false_alarms_per_frame
        return 1 / rate if rate else float("inf")


def sweep(assignment: Assignment, frames: int,
          steps: int = 50) -> list[Point]:
    """The whole curve, as operating points rather than as an integral.

    Thresholds are taken from the observed scores rather than from an even
    grid. An even grid wastes most of its points in ranges where no detection
    lives, and misses the sharp region near the top of the score distribution
    where the interesting trade actually happens.
    """
    if not assignment.scored or not assignment.positives:
        return []

    ordered = sorted(assignment.scored, key=lambda item: item[0], reverse=True)
    scores = [score for score, _ in ordered]

    # Evenly spaced in RANK, which is evenly spaced in "how many detections you
    # are accepting", the quantity the engineer is really moving.
    picks = sorted({int(i * (len(scores) - 1) / max(1, steps - 1))
                    for i in range(steps)})

    points: list[Point] = []
    hits = misses = 0
    cursor = 0
    for index in picks:
        while cursor <= index:
            _, is_hit = ordered[cursor]
            hits += is_hit
            misses += not is_hit
            cursor += 1
        accepted = hits + misses
        points.append(Point(
            threshold=scores[index],
            recall=hits / assignment.positives,
            precision=hits / accepted if accepted else 0.0,
            true_positives=hits,
            false_positives=misses,
            missed=assignment.positives - hits,
            frames=frames,
        ))
    return points


def threshold_for_recall(points: list[Point], target: float) -> Point | None:
    """The loosest threshold reaching a target recall, or None if unreachable.

    LOOSEST, not the first one found. Several thresholds may reach the target
    and the highest of them is the one that costs the fewest false alarms,
    which is the point anybody would actually choose.

    None is returned rather than the closest available, because "you cannot get
    there from here" is the answer, and a nearest match would be read as though
    the target had been met.
    """
    reaching = [p for p in points if p.recall >= target]
    return max(reaching, key=lambda p: p.threshold) if reaching else None


def best_recall(points: list[Point]) -> float:
    """The most that could be caught at any threshold.

    The ceiling. If this is below the target, no threshold choice fixes it and
    the answer is a different sensor, not a different number.
    """
    return max((p.recall for p in points), default=0.0)
