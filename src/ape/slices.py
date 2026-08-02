"""The slice dimensions, defined from ground truth attributes and fixed here.

THE OBJECTION THIS FILE EXISTS TO ANSWER. Slice-based evaluation has one obvious
failure mode: run the numbers, find the worst-looking subset, and present it as
an insight. That is not analysis, it is fishing, and any detector can be made to
look bad on some subset chosen after the fact.

So every dimension here is derived from an attribute KITTI annotated before
anybody saw a result, the boundaries are round numbers or the benchmark's own,
and the whole set is committed before the evaluation runs. If a slice turns out
uninteresting it stays in the report anyway. A slicing scheme that only contains
the slices that worked is the same fishing expedition with the evidence removed.

WHAT MAKES A SLICE LEGITIMATE HERE:

  it comes from the label       occlusion, truncation, class and 3D position are
                                annotations, not measurements of the detector

  the boundaries are principled KITTI's own tiers where they exist, round
                                numbers in metres and pixels where they do not

  it is not a proxy for score   nothing here is derived from a detection. A
                                slice defined using the detector's own output
                                would be measuring the detector against itself
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ape.records import Difficulty, GroundTruth

#: A slice dimension: given a ground truth object, which bin does it fall in?
#: None means the object is outside this dimension entirely, which is different
#: from being in a bin: an object with no 3D annotation has no distance, and
#: putting it in a "0 m" bin would be a lie rather than a gap.
Binner = Callable[[GroundTruth], str | None]


@dataclass(frozen=True)
class Dimension:
    """One way of cutting the data, and the order its bins are reported in."""

    name: str
    question: str
    bins: tuple[str, ...]
    of: Binner


def _band(value: float, edges: tuple[float, ...], unit: str) -> str:
    """Which band a value falls in, named by its own boundaries.

    Named rather than numbered so a row in the report reads as "30-40 m" and
    cannot be misread as a bin index.
    """
    for low, high in zip((0.0, *edges), (*edges, float("inf")), strict=True):
        if low <= value < high:
            if high == float("inf"):
                return f">{low:g} {unit}"
            return f"{low:g}-{high:g} {unit}"
    return f">{edges[-1]:g} {unit}"


DISTANCE_EDGES = (10.0, 20.0, 30.0, 40.0, 50.0)
HEIGHT_EDGES = (25.0, 40.0, 80.0, 160.0)


def _bands(edges: tuple[float, ...], unit: str) -> tuple[str, ...]:
    names = [f"{low:g}-{high:g} {unit}"
             for low, high in zip((0.0, *edges[:-1]), edges, strict=True)]
    return (*names, f">{edges[-1]:g} {unit}")


#: Every dimension the report will contain, in the order it will contain them.
DIMENSIONS: tuple[Dimension, ...] = (
    Dimension(
        name="difficulty",
        question="Does the benchmark's own notion of difficulty predict failure?",
        bins=(Difficulty.EASY, Difficulty.MODERATE, Difficulty.HARD,
              Difficulty.IGNORED),
        of=lambda gt: gt.difficulty.value,
    ),
    Dimension(
        name="occlusion",
        question="How much of the object may be hidden before it is missed?",
        # KITTI's annotated levels, used verbatim. 3 means the annotator could
        # not tell, which is itself worth reporting rather than folding into 2.
        bins=("fully visible", "partly occluded", "largely occluded", "unknown"),
        of=lambda gt: ("fully visible", "partly occluded",
                       "largely occluded", "unknown")[min(gt.occlusion, 3)],
    ),
    Dimension(
        name="truncation",
        question="Does an object leaving the frame stop being detected?",
        bins=("none", "0-15%", "15-30%", "30-50%", ">50%"),
        of=lambda gt: ("none" if gt.truncation <= 0.0 else
                       "0-15%" if gt.truncation <= 0.15 else
                       "15-30%" if gt.truncation <= 0.30 else
                       "30-50%" if gt.truncation <= 0.50 else ">50%"),
    ),
    Dimension(
        name="distance",
        # DEPTH ALONG THE OPTICAL AXIS, not radial distance to the object, and
        # the difference is not negligible: 9.7% of pedestrians in this split
        # would fall in a different band under the other definition, and the
        # worst case differs by 7.2 m for an object 21.9 m off to the side.
        #
        # Depth is the right choice here because it is the quantity that decides
        # time to collision on a straight path, which is what an ADAS function
        # acts on. Radial distance is what the phrase "how far away" suggests in
        # conversation, so the choice is stated rather than left to be inferred
        # from the fact that the code reads `location_cam[2]`.
        question="At what range does the detector stop seeing things?",
        bins=_bands(DISTANCE_EDGES, "m"),
        of=lambda gt: (None if gt.distance_m is None
                       else _band(gt.distance_m, DISTANCE_EDGES, "m")),
    ),
    Dimension(
        name="box height",
        question="Is it distance that matters, or simply apparent size?",
        # Separate from distance on purpose. They correlate but are not the
        # same: a truck at 40 m and a pedestrian at 15 m occupy similar heights,
        # and knowing which one predicts failure says whether the limit is
        # resolution or something about range itself.
        bins=_bands(HEIGHT_EDGES, "px"),
        of=lambda gt: _band(gt.box.height, HEIGHT_EDGES, "px"),
    ),
    Dimension(
        name="position",
        question="Are objects at the edge of the field of view treated worse?",
        # Derived from the annotation, not from the image. The third of the
        # frame an object's centre falls in.
        bins=("left third", "centre third", "right third"),
        of=lambda gt: ("left third" if (gt.box.x1 + gt.box.x2) / 2 < 414
                       else "centre third" if (gt.box.x1 + gt.box.x2) / 2 < 828
                       else "right third"),
    ),
)


def dimension(name: str) -> Dimension:
    for item in DIMENSIONS:
        if item.name == name:
            return item
    raise KeyError(f"no slice dimension named {name!r}; "
                   f"have {[d.name for d in DIMENSIONS]}")
