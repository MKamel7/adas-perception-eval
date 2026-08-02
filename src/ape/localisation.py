"""Did the detector not see it, or see it and box it badly?

THE DISTINCTION THIS MODULE EXISTS TO MAKE. Everything else in the project
counts a miss as a miss. But at IoU 0.5 those are two different failures with
two different fixes, and reporting them as one number tells an engineer to work
on the wrong thing:

  not detected           nothing was emitted anywhere near the object. The fix
                         is recall: a better backbone, more training data of
                         that kind, a different sensor.

  detected, mislocated   a box was emitted on the object and did not overlap
                         enough to count. The fix is box regression, anchor
                         geometry or the threshold convention itself, and the
                         detector already knows the object is there.

The second is much cheaper to fix and, for a safety argument, much less
alarming: a system that knows something is present at roughly the right place
can still brake for it. The demo scene made this visible one frame at a time
by printing the achieved overlap on each missed object. This quantifies it.

IT ALSO ANSWERS A LIMITATION THE README HAD BEEN CARRYING. KITTI scores Car at
IoU 0.7 and this project reported everything at 0.5, so the figures were not
comparable with the KITTI leaderboard. Sweeping the threshold gives the 0.7
number directly, and the shape of the sweep is the localisation diagnostic: a
detector whose AP collapses between 0.5 and 0.7 is localising loosely, one whose
AP is already low at 0.3 is not finding things at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from ape.match import Predicate, partition
from ape.records import Detection, GroundTruth

#: The thresholds swept. 0.7 is KITTI's own for Car, 0.5 is COCO's default and
#: what this project reports, 0.3 is loose enough that anything still missed at
#: it was genuinely not found.
THRESHOLDS = (0.3, 0.5, 0.7)

#: Below this, a detection is not "on" the object in any useful sense. Used to
#: separate a mislocated box from a coincidental overlap with something else.
TOUCHING = 0.1


@dataclass(frozen=True)
class Diagnosis:
    """Why the objects that were not found at the tight threshold were missed."""

    label: str
    tight: float
    loose: float
    total: int
    found_tight: int
    #: Missed at the tight threshold but overlapping a detection by at least
    #: the loose one: the detector saw it and boxed it badly.
    mislocated: int
    #: Nothing worth calling a detection anywhere near it.
    unseen: int

    @property
    def missed(self) -> int:
        return self.mislocated + self.unseen

    @property
    def mislocation_share(self) -> float:
        """Of everything missed, the fraction that was actually a box problem.

        The number that decides where effort goes. High means the detector
        already knows; low means it does not.
        """
        return self.mislocated / self.missed if self.missed else float("nan")


def diagnose(detections: dict[str, list[Detection]],
             truth: dict[str, list[GroundTruth]],
             label: str, neutral: frozenset[str],
             tight: float = 0.5, loose: float = 0.3,
             counts_when: Predicate | None = None) -> Diagnosis:
    """Split the misses at `tight` into mislocated and unseen.

    Greedy assignment at the tight threshold first, exactly as the scoring does,
    so "missed" here means the same thing it means everywhere else in the
    project. Only then is each unclaimed object asked whether any detection came
    close, which is a question about the detector rather than about the score.
    """
    total = found = mislocated = unseen = 0

    for frame_id, truths in truth.items():
        counts, _ = partition(truths, label, neutral, counts_when)
        if not counts:
            continue
        total += len(counts)

        mine = [d for d in detections.get(frame_id, []) if d.label == label]
        claimed: set[int] = set()
        for detection in sorted(mine, key=lambda d: d.score, reverse=True):
            best, index = 0.0, -1
            for i, candidate in enumerate(counts):
                if i in claimed:
                    continue
                overlap = detection.box.iou(candidate.box)
                if overlap > best:
                    best, index = overlap, i
            if index >= 0 and best >= tight:
                claimed.add(index)
        found += len(claimed)

        for i, candidate in enumerate(counts):
            if i in claimed:
                continue
            # The best any detection managed on this object, whether or not
            # that detection was used elsewhere. A box that was assigned to a
            # neighbour still proves the detector saw something here.
            nearest = max((d.box.iou(candidate.box) for d in mine), default=0.0)
            if nearest >= loose and nearest >= TOUCHING:
                mislocated += 1
            else:
                unseen += 1

    return Diagnosis(label=label, tight=tight, loose=loose, total=total,
                     found_tight=found, mislocated=mislocated, unseen=unseen)
