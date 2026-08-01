"""KITTI adapter: labels, calibration, and the projection that proves them consistent.

This is the only module in the project that knows what a KITTI file looks like.
Everything downstream sees `GroundTruth` records and cannot tell which dataset
produced them, which is what makes the sim-to-real comparison meaningful later.

WHY CALIBRATION IS HERE AT ALL, since a 2D detection benchmark does not obviously
need it. Two reasons, and the second is the important one.

The dull reason: distance is the slice that matters most in ADAS, and it is only
available from the 3D annotation.

The real reason: KITTI gives, for the same object, both a 3D position in camera
coordinates AND a 2D box in the image. Those two are redundant, and redundancy is
testable. Project the 3D centre through the calibration and it must land inside
the 2D box. Frame conventions are where this quietly goes wrong, and this is the
cheapest available guard against it.

HOW STRONG THAT GUARD ACTUALLY IS was measured rather than assumed, by breaking
the calibration six ways and seeing which breakages it noticed
(`tests/test_projection_sensitivity.py`):

  caught: a flipped sign, u and v swapped, a negated baseline translation
  MISSED: rectification skipped entirely, R0_rect transposed

The misses are not a bug to fix. KITTI's R0_rect is under one degree from
identity, so dropping it moves a projected point by a small fraction of a box,
which is well inside the slack that "the centre is somewhere in the box" allows.
The 3D-to-2D redundancy does not resolve a rotation that small. Rectification is
therefore proven to be applied by a separate test using a synthetic quarter turn,
because the real data cannot show it.

The first draft of this docstring claimed the check caught a dropped
rectification. It does not, and the sweep is what said so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ape.records import Box2D, GroundTruth

#: KITTI's label file columns, in order. Written down because the format is
#: positional and undocumented inside the files themselves.
#:
#:   type truncated occluded alpha  x1 y1 x2 y2  h w l  x y z  rotation_y
_FIELDS = 15

#: Objects KITTI annotates but excludes from scoring. Kept and marked rather
#: than silently dropped: a detector firing on a DontCare region is not a false
#: positive, and a pipeline that does not know that reports inflated FP counts.
IGNORE_CLASSES = frozenset({"DontCare"})


class CalibrationError(ValueError):
    """A calibration file that cannot be trusted, as opposed to one that is absent."""


@dataclass(frozen=True)
class Calibration:
    """The camera-2 projection and rectification for one frame.

    Only what a 2D pipeline actually needs is parsed. `P2` is the projection for
    the left colour camera, which is the camera KITTI's 2D benchmark uses, and
    `R0_rect` is the rectifying rotation that must be applied first.
    """

    #: 3x4 projection matrix for the rectified left colour camera.
    p2: tuple[tuple[float, ...], ...]
    #: 3x3 rectifying rotation.
    r0_rect: tuple[tuple[float, ...], ...]

    def project_cam_to_image(self, point_cam: tuple[float, float, float]
                             ) -> tuple[float, float]:
        """Project a point in CAMERA coordinates to pixels.

        The label's `location` is already in camera coordinates, so the velodyne
        transform is deliberately not involved. Rectification is applied first,
        because P2 is defined on rectified coordinates and applying it to
        unrectified ones is the classic silent error: everything still projects,
        just slightly wrong, and no exception is ever raised.
        """
        rectified = tuple(
            sum(self.r0_rect[row][col] * point_cam[col] for col in range(3))
            for row in range(3)
        )
        u, v, w = (
            sum(self.p2[row][col] * rectified[col] for col in range(3))
            + self.p2[row][3]
            for row in range(3)
        )

        if abs(w) < 1e-9:
            raise CalibrationError(
                f"point {point_cam} projects to a vanishing depth, w={w}")
        return u / w, v / w


def parse_calibration(text: str) -> Calibration:
    """Read a KITTI calib file. Raises rather than defaulting a missing matrix.

    Defaulting would produce a pipeline that runs on a broken calibration and
    reports distances that are confidently wrong, which is worse than stopping.
    """
    values: dict[str, list[float]] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        try:
            values[key.strip()] = [float(v) for v in rest.split()]
        except ValueError as bad:
            raise CalibrationError(f"{key.strip()} is not numeric") from bad

    for key, size in (("P2", 12), ("R0_rect", 9)):
        if key not in values:
            raise CalibrationError(f"calibration has no {key}")
        if len(values[key]) != size:
            raise CalibrationError(
                f"{key} has {len(values[key])} values, expected {size}")

    p = values["P2"]
    r = values["R0_rect"]
    return Calibration(
        p2=tuple(tuple(p[row * 4:row * 4 + 4]) for row in range(3)),
        r0_rect=tuple(tuple(r[row * 3:row * 3 + 3]) for row in range(3)),
    )


def parse_label_line(frame_id: str, line: str) -> GroundTruth | None:
    """One annotation, or None for a class the benchmark does not score."""
    parts = line.split()
    if len(parts) < _FIELDS:
        raise ValueError(
            f"{frame_id}: label has {len(parts)} fields, expected at least {_FIELDS}")

    label = parts[0]
    if label in IGNORE_CLASSES:
        return None

    truncation = float(parts[1])
    occlusion = int(float(parts[2]))
    box = Box2D(*(float(v) for v in parts[4:8]))
    dims = (float(parts[8]), float(parts[9]), float(parts[10]))
    location = (float(parts[11]), float(parts[12]), float(parts[13]))

    return GroundTruth(
        frame_id=frame_id,
        label=label,
        box=box,
        occlusion=occlusion,
        truncation=truncation,
        # z in camera coordinates is depth along the optical axis, which is the
        # quantity an ADAS engineer means by "how far away was it".
        distance_m=location[2],
        location_cam=location,
        dimensions_m=dims,
    )


def load_labels(path: Path) -> list[GroundTruth]:
    """All scored annotations for one frame."""
    frame_id = path.stem
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = parse_label_line(frame_id, line)
        if record is not None:
            out.append(record)
    return out


def load_calibration(path: Path) -> Calibration:
    return parse_calibration(path.read_text(encoding="utf-8"))


def frame_ids(label_dir: Path) -> list[str]:
    """Every frame with an annotation, sorted, so a run is reproducible."""
    return sorted(p.stem for p in label_dir.glob("*.txt"))
