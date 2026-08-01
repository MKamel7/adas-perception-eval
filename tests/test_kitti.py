"""The KITTI adapter, and the calibration check that makes it trustworthy.

The test worth reading is `test_projecting_the_3d_centre_lands_inside_the_2d_box`.
Everything else here is parsing.

KITTI annotates the same object twice, in different spaces: a 3D position in
camera coordinates and a 2D box in the image. That redundancy is free evidence.
Project one into the other and they must agree, so a transposed matrix, a dropped
rectification or a flipped sign shows up immediately against real data instead of
surviving into the distance slices, where it would silently move every object a
few metres and nobody would ever see it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ape.kitti import (
    Calibration,
    CalibrationError,
    frame_ids,
    load_calibration,
    load_labels,
    parse_calibration,
    parse_label_line,
)
from ape.records import Difficulty

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_frames() -> list[str]:
    return sorted(p.stem for p in (FIXTURES / "label_2").glob("*.txt"))


# --- parsing -----------------------------------------------------------------
def test_a_label_line_parses_into_a_record() -> None:
    line = ("Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 "
            "1.89 0.48 1.20 1.84 1.47 8.41 0.01")

    gt = parse_label_line("000000", line)

    assert gt is not None
    assert gt.label == "Pedestrian"
    assert gt.box.x1 == pytest.approx(712.40)
    assert gt.box.y2 == pytest.approx(307.92)
    assert gt.occlusion == 0
    assert gt.truncation == pytest.approx(0.0)
    assert gt.distance_m == pytest.approx(8.41)


def test_dontcare_regions_are_dropped_rather_than_scored() -> None:
    """A detector firing on a DontCare region is not a false positive.

    KITTI marks areas it declined to annotate. Counting them would inflate the
    false-positive rate against every detector equally, which is worse than
    useless: it would look like a real result.
    """
    line = "DontCare -1 -1 -10 500.0 170.0 590.0 190.0 -1 -1 -1 -1000 -1000 -1000 -10"

    assert parse_label_line("000000", line) is None


def test_a_truncated_label_line_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError, match="expected at least"):
        parse_label_line("000000", "Car 0.00 0 -1.5 100 100 200 200")


def test_blank_lines_in_a_label_file_are_skipped(tmp_path: Path) -> None:
    """KITTI files end with a newline, and some carry blank lines mid-file.

    Without this the trailing empty string reaches the parser and every frame
    raises on a field-count error, which would be caught immediately. The real
    risk is the opposite: a lenient parser that turns a blank line into a
    zero-area ground truth nobody can ever detect.
    """
    label = tmp_path / "000000.txt"
    label.write_text(
        "Car 0.00 0 -1.5 100.0 100.0 200.0 200.0 1.5 1.6 4.0 1.0 1.5 20.0 -1.5\n"
        "\n"
        "   \n"
        "Car 0.00 0 -1.5 300.0 100.0 400.0 200.0 1.5 1.6 4.0 5.0 1.5 20.0 -1.5\n",
        encoding="utf-8")

    records = load_labels(label)

    assert len(records) == 2
    assert all(r.box.area > 0 for r in records)


def test_frame_ids_are_sorted_so_a_run_is_reproducible() -> None:
    """Directory iteration order is not guaranteed, and an evaluation whose
    frame order shifts between runs cannot be compared against its own
    earlier results."""
    ids = frame_ids(FIXTURES / "label_2")

    assert ids == sorted(ids)
    assert len(ids) == 20
    assert all("." not in i for i in ids), "an id kept its extension"


# --- difficulty --------------------------------------------------------------
@pytest.mark.parametrize("height,occlusion,truncation,expected", [
    (60, 0, 0.00, Difficulty.EASY),
    (60, 1, 0.00, Difficulty.MODERATE),   # occluded, so not easy
    (30, 0, 0.00, Difficulty.MODERATE),   # too short for easy
    (30, 2, 0.00, Difficulty.HARD),
    (60, 0, 0.40, Difficulty.HARD),       # truncation pushes it down
    (20, 0, 0.00, Difficulty.IGNORED),    # below every tier
    (60, 3, 0.00, Difficulty.IGNORED),    # unknown occlusion
])
def test_difficulty_follows_kittis_published_thresholds(
        height: float, occlusion: int, truncation: float,
        expected: Difficulty) -> None:
    """Used rather than invented, and that is the point.

    A slicing scheme the benchmark published before anybody saw these results
    cannot be accused of having been chosen to flatter them, which is the first
    objection raised about slice-based evaluation.
    """
    line = (f"Car {truncation:.2f} {occlusion} -1.5 "
            f"100.0 100.0 200.0 {100.0 + height:.1f} "
            f"1.5 1.6 4.0 1.0 1.5 20.0 -1.5")

    gt = parse_label_line("000000", line)

    assert gt is not None
    assert gt.difficulty is expected


# --- calibration -------------------------------------------------------------
def test_calibration_parses_the_two_matrices_a_2d_pipeline_needs() -> None:
    calib = load_calibration(FIXTURES / "calib" / f"{fixture_frames()[0]}.txt")

    assert len(calib.p2) == 3 and len(calib.p2[0]) == 4
    assert len(calib.r0_rect) == 3 and len(calib.r0_rect[0]) == 3


@pytest.mark.parametrize("text,reason", [
    ("R0_rect: 1 0 0 0 1 0 0 0 1", "no P2"),
    ("P2: 1 2 3\nR0_rect: 1 0 0 0 1 0 0 0 1", "P2 has 3 values"),
    ("P2: a b c\nR0_rect: 1 0 0 0 1 0 0 0 1", "not numeric"),
])
def test_a_calibration_that_cannot_be_trusted_is_refused(text: str,
                                                         reason: str) -> None:
    """Refused rather than defaulted.

    A defaulted matrix produces a pipeline that runs happily and reports
    distances that are confidently wrong, and every slice downstream inherits
    the error without any sign that something failed.
    """
    with pytest.raises(CalibrationError):
        parse_calibration(text)


# --- the one that matters ----------------------------------------------------
def test_projecting_the_3d_centre_lands_inside_the_2d_box() -> None:
    """The two annotations of the same object must agree.

    Checked across every object in the fixture rather than one hand-picked
    example, because a projection error can be small enough to survive a single
    generous case.

    The centre is allowed to fall slightly outside a truncated object's box,
    since the box is clipped at the image edge while the 3D centre is not, so
    those are skipped rather than fudged with a tolerance.
    """
    checked = 0
    for frame in fixture_frames():
        calib = load_calibration(FIXTURES / "calib" / f"{frame}.txt")
        for gt in load_labels(FIXTURES / "label_2" / f"{frame}.txt"):
            if gt.location_cam is None or gt.truncation > 0.05:
                continue
            x, y, z = gt.location_cam
            if z < 1.0:
                continue
            height = gt.dimensions_m[0] if gt.dimensions_m else 0.0
            # KITTI's location is the centre of the BOTTOM face, so half the
            # object's height is added to reach the centroid. Getting this wrong
            # is itself a classic error and this test would catch it.
            u, v = calib.project_cam_to_image((x, y - height / 2, z))

            assert gt.box.contains(u, v), (
                f"{frame} {gt.label}: 3D centre projects to ({u:.1f}, {v:.1f}) "
                f"which is outside its 2D box {gt.box}"
            )
            checked += 1

    assert checked >= 20, f"only {checked} objects checked, the fixture proves little"


def test_a_sign_flip_in_the_calibration_is_caught() -> None:
    """The guard on the guard.

    A consistency check nobody has watched fail is an assumption. This flips one
    sign in the projection and confirms the objects stop landing in their boxes,
    which is what makes the test above evidence rather than decoration.
    """
    frame = fixture_frames()[0]
    good = load_calibration(FIXTURES / "calib" / f"{frame}.txt")
    flipped = Calibration(
        p2=tuple(tuple(-v if col == 0 else v for col, v in enumerate(row))
                 for row in good.p2),
        r0_rect=good.r0_rect,
    )

    escaped = 0
    for gt in load_labels(FIXTURES / "label_2" / f"{frame}.txt"):
        if gt.location_cam is None or gt.truncation > 0.05:
            continue
        x, y, z = gt.location_cam
        if z < 1.0:
            continue
        height = gt.dimensions_m[0] if gt.dimensions_m else 0.0
        u, v = flipped.project_cam_to_image((x, y - height / 2, z))
        if not gt.box.contains(u, v):
            escaped += 1

    assert escaped > 0, "a flipped sign changed nothing, so the check proves nothing"


def test_a_point_on_the_camera_plane_is_refused_rather_than_dividing_by_zero() -> None:
    """A plain pinhole, so the depth that vanishes is obviously z = 0.

    Deliberately not KITTI's own calibration: its P2 carries a small baseline
    translation, so a point at z = 0 still projects at about 3 mm of depth and
    would never reach this guard. Reaching for the real matrix here would mean
    inverting rectification to find the depth that does vanish, which is more
    arithmetic than the thing being tested.
    """
    pinhole = Calibration(
        p2=((700.0, 0.0, 600.0, 0.0),
            (0.0, 700.0, 190.0, 0.0),
            (0.0, 0.0, 1.0, 0.0)),
        r0_rect=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )

    with pytest.raises(CalibrationError, match="vanishing depth"):
        pinhole.project_cam_to_image((1.0, 1.0, 0.0))


# --- the fixture itself ------------------------------------------------------
def test_the_fixture_is_big_enough_to_mean_something() -> None:
    frames = fixture_frames()

    assert len(frames) >= 20
    assert all((FIXTURES / "calib" / f"{f}.txt").exists() for f in frames), (
        "a frame has labels but no calibration"
    )
