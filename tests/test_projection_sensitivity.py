"""What the projection check actually catches, and what it does not.

`test_projecting_the_3d_centre_lands_inside_the_2d_box` is the guard that the
whole distance slice depends on. This module answers the question that guard
cannot answer about itself: **which errors would it survive?**

Each case below breaks the calibration in a specific, plausible way and records
whether the check notices. Two of them it does not notice, and that is written
down here rather than discovered later by someone trusting the docstring.

This is the same lesson the virtual production cell taught: 100% branch coverage
said the calibration code was fully tested, and the mutation sweep is what
revealed that two realistic bugs pass straight through it. Coverage measures
which lines ran, not which situations were imagined.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ape.kitti import Calibration, frame_ids, load_calibration, load_labels

FIXTURES = Path(__file__).parent / "fixtures"
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def escapes_under(mutate: Callable[[Calibration], Calibration]) -> tuple[int, int]:
    """How many objects leave their 2D box once the calibration is broken."""
    escaped = total = 0
    for frame in frame_ids(FIXTURES / "label_2"):
        calib = mutate(load_calibration(FIXTURES / "calib" / f"{frame}.txt"))
        for gt in load_labels(FIXTURES / "label_2" / f"{frame}.txt"):
            if gt.location_cam is None or gt.truncation > 0.05:
                continue
            x, y, z = gt.location_cam
            if z < 1.0:
                continue
            height = gt.dimensions_m[0] if gt.dimensions_m else 0.0
            u, v = calib.project_cam_to_image((x, y - height / 2, z))
            total += 1
            if not gt.box.contains(u, v):
                escaped += 1
    return escaped, total


def flip_column(index: int) -> Callable[[Calibration], Calibration]:
    def mutate(c: Calibration) -> Calibration:
        return Calibration(
            p2=tuple(tuple(-v if i == index else v for i, v in enumerate(row))
                     for row in c.p2),
            r0_rect=c.r0_rect)
    return mutate


def swap_uv_rows(c: Calibration) -> Calibration:
    return Calibration(p2=(c.p2[1], c.p2[0], c.p2[2]), r0_rect=c.r0_rect)


def drop_rectification(c: Calibration) -> Calibration:
    return Calibration(p2=c.p2, r0_rect=IDENTITY)


def transpose_rectification(c: Calibration) -> Calibration:
    return Calibration(p2=c.p2, r0_rect=tuple(zip(*c.r0_rect, strict=True)))


@pytest.mark.parametrize("name,mutate", [
    ("a sign flipped in the projection", flip_column(0)),
    ("u and v swapped", swap_uv_rows),
    ("the baseline translation negated", flip_column(3)),
])
def test_the_check_catches_gross_frame_errors(
        name: str, mutate: Callable[[Calibration], Calibration]) -> None:
    """These are the errors the check is evidence against.

    A guard nobody has watched fail is an assumption, so each of these is
    watched failing rather than assumed to fail.
    """
    escaped, _ = escapes_under(mutate)

    assert escaped > 0, f"{name} changed nothing, so the check proves nothing"


def test_the_unmutated_calibration_puts_every_object_in_its_box() -> None:
    """The control. Without it, a mutation test can pass because the check is
    broken rather than because the mutation was caught."""
    escaped, total = escapes_under(lambda c: c)

    assert escaped == 0
    assert total >= 20, f"only {total} objects exercised, the fixture proves little"


@pytest.mark.parametrize("name,mutate", [
    ("rectification skipped entirely", drop_rectification),
    ("R0_rect transposed", transpose_rectification),
])
def test_the_check_is_blind_to_rectification_errors(
        name: str, mutate: Callable[[Calibration], Calibration]) -> None:
    """A KNOWN LIMIT, asserted so it cannot quietly change.

    KITTI's R0_rect is less than one degree from identity. Skipping it moves a
    projected point by a small fraction of a box, which is far inside the slack
    that "the centre is somewhere in the box" allows. The redundancy between the
    3D and 2D annotations simply does not resolve a rotation that small, and no
    threshold on this check can separate the two: the worst offset with
    rectification dropped is actually SMALLER than the worst offset without any
    mutation at all.

    So this is a property of the data, not a fix waiting to be written. It is
    asserted in the failing direction on purpose: if a future dataset or a
    changed fixture makes rectification observable here, this test fails and the
    limit gets revisited instead of being carried forward as folklore.
    """
    escaped, _ = escapes_under(mutate)

    assert escaped == 0, (
        f"{name} is now detectable, so the documented limit is stale and the "
        f"projection check is stronger than it claims"
    )


def test_rectification_is_nevertheless_applied() -> None:
    """Since real data cannot show it, prove it directly.

    A large synthetic rotation makes the effect unmissable, which closes the gap
    the two tests above leave open: that R0_rect might be parsed, stored, and
    never actually multiplied in.
    """
    quarter_turn = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    p2 = ((700.0, 0.0, 600.0, 0.0), (0.0, 700.0, 190.0, 0.0), (0.0, 0.0, 1.0, 0.0))
    point = (2.0, 1.0, 20.0)

    unrectified = Calibration(p2=p2, r0_rect=IDENTITY).project_cam_to_image(point)
    rectified = Calibration(p2=p2, r0_rect=quarter_turn).project_cam_to_image(point)

    assert unrectified != rectified
