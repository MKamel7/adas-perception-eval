"""The canonical records everything downstream speaks.

THE SEAM, and it is the reason this project can answer the question it is for.

Every dataset adapter emits `GroundTruth`, every detector adapter emits
`Detection`, and nothing after this module knows which dataset or which model
produced them. Matching, metrics, slicing and the report are written once.

That is not tidiness. The sim-to-real comparison is only possible BECAUSE the
same code can be pointed at KITTI and at Virtual KITTI 2 with no branch anywhere
in the measurement path. If the two were measured by even slightly different
code, any difference between them would be uninterpretable, which would destroy
the one experiment this project exists to run.

Same instinct as the process image in the virtual production cell: put the
contract in one place and make everything else agree with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Difficulty(StrEnum):
    """KITTI's own difficulty tiers.

    Used rather than invented because the benchmark DEFINES them, from occlusion,
    truncation and box height. A slicing scheme somebody else published before
    seeing these results cannot be accused of having been chosen to flatter them,
    and that objection is the first one a reviewer raises about slice-based
    evaluation.
    """

    EASY = "easy"
    MODERATE = "moderate"
    HARD = "hard"
    #: Below every tier's thresholds. KITTI excludes these from its benchmark
    #: entirely, and they are kept here rather than dropped because "what the
    #: benchmark refuses to score" is itself a slice worth looking at.
    IGNORED = "ignored"


@dataclass(frozen=True)
class Box2D:
    """An axis-aligned image-space box, in pixels, left-top to right-bottom."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def iou(self, other: Box2D) -> float:
        """Intersection over union.

        Written here rather than pulled from a library because it is three lines
        and it is the single arithmetic operation the entire evaluation rests on.
        A subtly wrong IoU produces plausible metrics that are quietly wrong,
        which is the worst failure mode available to this project.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class GroundTruth:
    """One annotated object, dataset-agnostic.

    The 3D fields are optional because not every dataset has them, and they are
    carried rather than dropped because DISTANCE is the slice that matters most
    in ADAS and it is only derivable from them.
    """

    frame_id: str
    label: str
    box: Box2D
    #: 0 fully visible, 1 partly occluded, 2 largely occluded, 3 unknown.
    occlusion: int
    #: Fraction of the object leaving image boundaries, 0.0 to 1.0.
    truncation: float
    #: Metres along the camera's optical axis. None when the dataset has no 3D.
    distance_m: float | None = None
    #: Object centre in camera coordinates, metres. Used to verify projection.
    location_cam: tuple[float, float, float] | None = None
    dimensions_m: tuple[float, float, float] | None = None

    @property
    def difficulty(self) -> Difficulty:
        """KITTI's published thresholds, applied verbatim.

        Height is the box height in pixels, and it is the reason a distant
        pedestrian and a near one are not the same test even though they carry
        the same class label.
        """
        h = self.box.height
        if h >= 40 and self.occlusion <= 0 and self.truncation <= 0.15:
            return Difficulty.EASY
        if h >= 25 and self.occlusion <= 1 and self.truncation <= 0.30:
            return Difficulty.MODERATE
        if h >= 25 and self.occlusion <= 2 and self.truncation <= 0.50:
            return Difficulty.HARD
        return Difficulty.IGNORED


@dataclass(frozen=True)
class Detection:
    """One predicted object, detector-agnostic."""

    frame_id: str
    label: str
    box: Box2D
    score: float
