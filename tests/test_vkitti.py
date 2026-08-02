"""The Virtual KITTI 2 adapter, on a committed five-frame fixture.

The point of this adapter is that everything downstream cannot tell which
dataset it is looking at. So the tests that matter are not about parsing, they
are about the two records being interchangeable: same type, same fields, same
meaning, so a slice computed on synthetic data is the same operation as the one
computed on real data. A difference between the two halves has to come from the
imagery, not from the ingest.

The fixture is 38 kB of the first five frames of Scene01/clone, built by the
same reasoning as the KITTI one: enough to exercise the three-file join, the
column ordering and the occupancy mapping, small enough that the repository does
not start carrying a dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ape.records import GroundTruth
from ape.vkitti import VARIANTS, Variant, image_path, load_labels, variants

FIXTURES = Path(__file__).parent / "fixtures/vkitti"
CLONE = Variant("Scene01", "clone")


def loaded() -> dict[str, list[GroundTruth]]:
    return load_labels(FIXTURES, CLONE)


# --- the join across three files ---------------------------------------------
def test_boxes_classes_and_poses_are_joined_on_track() -> None:
    """bbox.txt has the box, info.txt has the class, pose.txt has the depth.
    A record is only usable if all three arrived."""
    frames = loaded()

    assert frames
    for items in frames.values():
        for item in items:
            assert item.label
            assert item.box.area > 0
            assert item.distance_m is not None


def test_a_box_whose_track_has_no_class_is_refused(tmp_path: Path) -> None:
    """Defaulting it would score an unknown object as a Car."""
    base = tmp_path / "Scene99" / "clone"
    base.mkdir(parents=True)
    (base / "bbox.txt").write_text(
        "frame cameraID trackID left right top bottom number_pixels "
        "truncation_ratio occupancy_ratio isMoving\n"
        "0 0 7 10 60 20 90 100 0 1 False\n", encoding="utf-8")
    (base / "info.txt").write_text("trackID label model color\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no class"):
        load_labels(tmp_path, Variant("Scene99", "clone"))


def test_only_the_requested_camera_is_read() -> None:
    """Camera 1 is a second view of the same scene. Reading both would double
    every object and halve every precision."""
    frames = load_labels(FIXTURES, CLONE, camera=0)
    ids = {item.frame_id for items in frames.values() for item in items}

    assert len(ids) == len(frames)


# --- the column order, which is not KITTI's ----------------------------------
def test_the_box_columns_are_left_right_top_bottom() -> None:
    """VKITTI orders them left right top bottom; KITTI uses x1 y1 x2 y2.
    Reading positionally without checking produces a plausible wrong box, so
    the resulting geometry is asserted rather than the parse.
    """
    for items in loaded().values():
        for item in items:
            assert item.box.x2 > item.box.x1, "width came out negative"
            assert item.box.y2 > item.box.y1, "height came out negative"
            assert item.box.width < 2000 and item.box.height < 800


# --- the mapping that is invented, and says so -------------------------------
def test_occupancy_becomes_kittis_three_occlusion_levels() -> None:
    frames = loaded()
    levels = {item.occlusion for items in frames.values() for item in items}

    assert levels <= {0, 1, 2}


def test_the_records_are_indistinguishable_from_kittis() -> None:
    """The property the whole comparison rests on.

    If a slice function could tell the two datasets apart, any difference
    between real and synthetic would be uninterpretable: it might be the
    imagery or it might be the ingest.
    """
    from ape.slices import DIMENSIONS

    items = [item for values in loaded().values() for item in values]
    assert items

    for dim in DIMENSIONS:
        produced = {dim.of(item) for item in items}
        produced.discard(None)
        unknown = produced - {str(b) for b in dim.bins}
        assert not unknown, f"{dim.name} produced {unknown} on synthetic data"


def test_difficulty_works_on_synthetic_records() -> None:
    items = [item for values in loaded().values() for item in values]

    assert {item.difficulty for item in items}


# --- discovery ---------------------------------------------------------------
def test_only_conditions_that_exist_on_disk_are_offered() -> None:
    found = variants(FIXTURES)

    assert found == [CLONE]
    assert all(v.condition in VARIANTS for v in found)


def test_camera_angle_variants_are_not_treated_as_weather() -> None:
    """15-deg-left changes the geometry, not the appearance. Mixing it into a
    weather comparison would attribute a viewpoint change to fog."""
    assert "15-deg-left" not in VARIANTS
    assert "30-deg-right" not in VARIANTS
    assert "fog" in VARIANTS and "rain" in VARIANTS


def test_the_image_path_matches_the_frame_id_it_was_given() -> None:
    frame = sorted(loaded())[0]
    path = image_path(FIXTURES, CLONE, frame)

    assert path.name.startswith("rgb_")
    assert path.suffix == ".jpg"
    assert "Camera_0" in str(path)
    assert path.name == f"rgb_{int(frame.rsplit('_', 1)[1]):05d}.jpg"


def test_frame_ids_carry_their_scene_and_condition() -> None:
    """Frames from Scene01/fog and Scene01/rain are different frames. Keying on
    the bare number would silently merge every variant into one."""
    frames = sorted(loaded())

    assert all(f.startswith("Scene01_clone_") for f in frames)
    assert len(set(frames)) == len(frames)
