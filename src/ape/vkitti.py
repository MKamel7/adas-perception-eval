"""Virtual KITTI 2 adapter: the synthetic half, behind the same records.

Emits `GroundTruth`, exactly as `kitti.py` does, so every stage after ingest is
identical for real and synthetic data. That is not tidiness, it is what makes
the comparison mean anything: if the two halves were measured by even slightly
different code, a difference between them would be uninterpretable.

THE LIMITATION THAT MATTERS MOST, found before writing any of this and stated
here rather than in a footnote:

    VIRTUAL KITTI 2 CONTAINS NO PEDESTRIANS. Across all five scenes the
    annotated classes are Car (245 tracks), Van (22) and Truck (6). There are no
    people and no cyclists.

The headline finding of this project is that pedestrian detection collapses
beyond 30 metres. **That finding cannot be checked against this simulation at
all**, because the simulation has no pedestrians to be blind to. This is the
sim-to-real result before a single number is computed, and it is a more useful
one than a correlation would have been: a validation programme that relied on
this synthetic data would not merely have understated the pedestrian problem, it
would have had no way to see it.

So M5 compares what CAN be compared, Car degradation, and reports the absence as
the finding it is.

WHAT IS MAPPED, AND WHAT IS INVENTED. The box, truncation and distance come
straight across. Occlusion does not: KITTI annotates a three-level judgement and
Virtual KITTI 2 reports a continuous `occupancy_ratio`. The thresholds below are
mine, not either dataset's, and that is flagged wherever the occlusion slice is
compared across the two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ape.records import Box2D, GroundTruth

#: The re-rendered conditions. `clone` reproduces the original sequence and is
#: the baseline every other variant is compared against; the camera-angle
#: variants are excluded because they change the geometry rather than the
#: appearance, which is a different question.
VARIANTS = ("clone", "fog", "morning", "overcast", "rain", "sunset")

#: occupancy_ratio -> KITTI's occlusion levels. INVENTED HERE, because the two
#: datasets describe occlusion differently and something has to bridge them.
#: 0.9 and 0.5 are round numbers, not derived, and any occlusion comparison
#: across the two datasets inherits that arbitrariness and must say so.
OCCUPANCY_LEVELS = ((0.90, 0), (0.50, 1))


@dataclass(frozen=True)
class Variant:
    """One scene rendered under one condition."""

    scene: str
    condition: str

    @property
    def key(self) -> str:
        return f"{self.scene}/{self.condition}"


def _occlusion_from(occupancy: float) -> int:
    for threshold, level in OCCUPANCY_LEVELS:
        if occupancy >= threshold:
            return level
    return 2


def load_labels(root: Path, variant: Variant, camera: int = 0
                ) -> dict[str, list[GroundTruth]]:
    """Every annotated object in one scene-condition, keyed by frame id.

    Three files have to agree: `bbox.txt` for the 2D box, `info.txt` for the
    class of each track, and `pose.txt` for the 3D position that gives distance.
    They are joined on trackID, and a box whose track is missing from `info.txt`
    is refused rather than defaulted, because a defaulted class would quietly
    become a Car and be scored as one.
    """
    base = root / variant.scene / variant.condition

    classes: dict[str, str] = {}
    for line in (base / "info.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            classes[parts[0]] = parts[1]

    #: (frame, track) -> depth along the optical axis, in metres.
    depth: dict[tuple[str, str], float] = {}
    pose = base / "pose.txt"
    if pose.exists():
        header = pose.read_text(encoding="utf-8").splitlines()[0].split()
        z_index = header.index("camera_space_Z")
        for line in pose.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) > z_index and parts[1] == str(camera):
                depth[(parts[0], parts[2])] = float(parts[z_index])

    by_frame: dict[str, list[GroundTruth]] = {}
    for line in (base / "bbox.txt").read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split()
        if len(parts) < 10 or parts[1] != str(camera):
            continue
        frame, track = parts[0], parts[2]
        label = classes.get(track)
        if label is None:
            raise ValueError(
                f"{variant.key}: track {track} has a box but no class in "
                f"info.txt. Defaulting it would score an unknown object as a Car.")

        # left right top bottom, NOT the x1 y1 x2 y2 order KITTI uses. Reading
        # these positionally without checking the header is the obvious way to
        # get a plausible, wrong box.
        left, right, top, bottom = (float(v) for v in parts[3:7])
        truncation = float(parts[8])
        occupancy = float(parts[9])

        frame_id = f"{variant.scene}_{variant.condition}_{int(frame):05d}"
        by_frame.setdefault(frame_id, []).append(GroundTruth(
            frame_id=frame_id,
            label=label,
            box=Box2D(left, top, right, bottom),
            occlusion=_occlusion_from(occupancy),
            truncation=truncation,
            distance_m=depth.get((frame, track)),
        ))
    return by_frame


def image_path(root: Path, variant: Variant, frame_id: str,
               camera: int = 0) -> Path:
    """Where the rendered frame lives, given an id produced by `load_labels`."""
    number = int(frame_id.rsplit("_", 1)[1])
    return (root / variant.scene / variant.condition / "frames" / "rgb"
            / f"Camera_{camera}" / f"rgb_{number:05d}.jpg")


def variants(root: Path) -> list[Variant]:
    """Every scene-condition present on disk, in a deterministic order."""
    found = []
    for scene in sorted(p.name for p in root.glob("Scene*") if p.is_dir()):
        for condition in VARIANTS:
            if (root / scene / condition / "bbox.txt").exists():
                found.append(Variant(scene, condition))
    return found
