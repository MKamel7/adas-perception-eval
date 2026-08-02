"""How a COCO detector's classes line up with KITTI's, written as data.

THIS IS THE MOST DANGEROUS FILE IN THE PROJECT, which is why the mapping is data
and not a series of `if` statements buried in the matcher. Every number this
project reports depends on it, and a wrong entry here does not crash: it
produces a complete, plausible evaluation that is quietly measuring the wrong
thing. `person` to `Pedestrian` looks obvious and hides `Person_sitting`.

THREE SEPARATE IDEAS LIVE HERE, and conflating them is the usual mistake.

1. WHICH COCO CLASS BECOMES WHICH KITTI CLASS. A detector trained on COCO has
   never heard of a Tram. This is a translation, and where no honest translation
   exists the entry is absent rather than guessed.

2. WHICH GROUND TRUTH IS NEUTRAL. Not every annotation is a target or a
   background. KITTI's own benchmark says a detection landing on a Van must not
   count as a false positive when evaluating Car, and the same for
   Person_sitting when evaluating Pedestrian. These are KITTI's published rules,
   not a convenience invented here, which matters because "we excluded some
   objects" is otherwise indistinguishable from flattering the result.

3. WHAT CANNOT BE MEASURED HONESTLY AT ALL. See `Cyclist` below.

THE CYCLIST PROBLEM, stated rather than hidden. KITTI annotates a cyclist as ONE
box containing rider and bicycle together. A COCO detector emits TWO boxes, a
`person` and a `bicycle`, neither of which has the shape of KITTI's box. There
is no mapping that makes this comparison fair:

  map `bicycle` to Cyclist   the box covers the bike, not the rider, so IoU
                             against KITTI's box is poor and the resulting AP
                             measures box convention, not detection ability
  map `person` to Cyclist    every pedestrian becomes a false cyclist
  fuse the two               requires inventing a heuristic and putting it in
                             the measurement path, so the result would then
                             depend on a rule nobody else uses

So Cyclist is reported, with `bicycle` mapped to it, and labelled everywhere as
mapping-limited. Its number is a lower bound on the detector and an upper bound
on nothing. It is NOT included in the headline figure, and the reason is stated
in the report rather than left for a reader to work out.
"""

from __future__ import annotations

from types import MappingProxyType

#: COCO class name -> KITTI class name. Absent means "no honest translation":
#: a COCO detector has no Tram class, and inventing one from `train` would make
#: a rail vehicle on a street indistinguishable from a train on rails.
#:
#: `bus` maps to Truck because KITTI has no bus class and a bus is a large
#: goods-shaped vehicle in every geometric sense that matters to a 2D box. This
#: is a judgement call and it is written here so it can be argued with.
COCO_TO_KITTI: MappingProxyType[str, str] = MappingProxyType({
    "car": "Car",
    "truck": "Truck",
    "bus": "Truck",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
})

#: The classes actually scored, in the order they are reported.
#: Truck and Van are annotated by KITTI but are not part of its published
#: benchmark, and adding classes the benchmark does not score would make these
#: numbers incomparable to every published result.
EVALUATED = ("Car", "Pedestrian", "Cyclist")

#: Classes carried in the headline. Cyclist is excluded for the reason at the
#: top of this file, and excluding it is stated in the report rather than being
#: a silent omission.
HEADLINE = ("Car", "Pedestrian")

#: KITTI's OWN neutrality rules, quoted rather than invented.
#:
#: Evaluating Car, a detection on a Van is neither right nor wrong: the two are
#: genuinely similar and the benchmark declines to punish either answer. Same
#: for Person_sitting when evaluating Pedestrian. Without this the false
#: positive count is inflated for every detector equally, which is worse than
#: useless because it still looks like a result.
NEUTRAL_FOR: MappingProxyType[str, frozenset[str]] = MappingProxyType({
    "Car": frozenset({"Van"}),
    "Pedestrian": frozenset({"Person_sitting"}),
    "Cyclist": frozenset(),
})

#: Never a target and never a false positive, whatever is being evaluated.
#: KITTI marks regions it declined to annotate; a detector firing there has not
#: made a mistake.
ALWAYS_NEUTRAL = frozenset({"DontCare"})


class ClassMappingError(ValueError):
    """A mapping that cannot be trusted, as opposed to one that is incomplete."""


def to_kitti(coco_name: str) -> str | None:
    """The KITTI class for a COCO detection, or None if it is not scored.

    None rather than a fallback class. A detector firing on a `traffic light`
    is not making a claim about any KITTI object, and folding it into `Misc`
    would turn correct silence into a false positive.
    """
    return COCO_TO_KITTI.get(coco_name)


def neutral_labels(evaluated_class: str) -> frozenset[str]:
    """Ground truth labels that neither count for nor against this class."""
    if evaluated_class not in NEUTRAL_FOR:
        raise ClassMappingError(
            f"{evaluated_class!r} has no neutrality rule. Adding a class to "
            f"EVALUATED without deciding what is neutral for it would silently "
            f"treat every other annotation as background.")
    return NEUTRAL_FOR[evaluated_class] | ALWAYS_NEUTRAL
