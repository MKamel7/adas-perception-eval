"""The modules that had no tests: cache, slices, evaluate, report, vkitti.

These were carried by the fact that running the pipeline exercised them, which
is not the same as testing them: a pipeline run says the code did not crash on
one input, and says nothing about the inputs nobody tried. Coverage sat at 47%
with six modules at zero, and every one of them is in the path between the
detector and a number somebody would quote.

The detector itself is only partly covered here. Its arithmetic, the letterbox
transform and non-maximum suppression, is tested; the ONNX session is not,
because standing a 45 MB model up in CI to prove that onnxruntime returns
tensors would be testing onnxruntime.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ape.cache import CacheError, load_detections, processed_frames
from ape.detect import Letterbox, _nms
from ape.evaluate import evaluate
from ape.kitti import load_labels
from ape.records import Box2D, Detection, Difficulty, GroundTruth
from ape.report import render
from ape.slices import DIMENSIONS, dimension

FIXTURES = Path(__file__).parent / "fixtures"


def gt(label: str = "Car", **kwargs: object) -> GroundTruth:
    base: dict = dict(frame_id="f", label=label, box=Box2D(100, 100, 200, 200),
                      occlusion=0, truncation=0.0, distance_m=15.0)
    base.update(kwargs)
    return GroundTruth(**base)  # type: ignore[arg-type]


# --- the detection cache -----------------------------------------------------
def test_a_cache_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "model": "m.onnx", "frames": 2,
                    "score_threshold": 0.05, "first_frame": "000000",
                    "last_frame": "000001"}) + "\n"
        + json.dumps({"frame_id": "000000", "label": "Car", "score": 0.9,
                      "box": [1, 2, 3, 4]}) + "\n", encoding="utf-8")

    header, detections = load_detections(path)

    assert header.model == "m.onnx"
    assert header.frames == 2
    assert detections["000000"][0].box == Box2D(1, 2, 3, 4)


def test_a_cache_without_a_header_is_refused(tmp_path: Path) -> None:
    """A detection file with no record of which model or split produced it can
    still be evaluated, and the number cannot be attributed or reproduced,
    which is worse than no number."""
    path = tmp_path / "d.jsonl"
    path.write_text(json.dumps({"frame_id": "0", "label": "Car", "score": 1.0,
                                "box": [0, 0, 1, 1]}) + "\n", encoding="utf-8")

    with pytest.raises(CacheError, match="no header"):
        load_detections(path)


def test_an_empty_cache_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(CacheError, match="empty"):
        load_detections(path)


def test_a_truncated_cache_fails_on_the_line_it_was_cut_at(tmp_path: Path) -> None:
    """The whole reason for JSONL over a pickle. A half-written cache must not
    load as a shorter evaluation nobody notices."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "model": "m", "frames": 1,
                    "score_threshold": 0.05, "first_frame": "a",
                    "last_frame": "a"}) + "\n"
        + '{"frame_id": "a", "label": "Car", "sco', encoding="utf-8")

    with pytest.raises(CacheError, match="not valid JSON"):
        load_detections(path)


def test_blank_lines_in_a_cache_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "model": "m", "frames": 1,
                    "score_threshold": 0.05, "first_frame": "a",
                    "last_frame": "a"}) + "\n\n\n"
        + json.dumps({"frame_id": "a", "label": "Car", "score": 0.5,
                      "box": [0, 0, 2, 2]}) + "\n", encoding="utf-8")

    _, detections = load_detections(path)

    assert len(detections["a"]) == 1


# --- resume, and the distinction it rests on ---------------------------------
def written(path: Path, *rows: dict) -> Path:
    header = {"kind": "header", "model": "m.onnx", "frames": 3,
              "score_threshold": 0.05, "first_frame": "a", "last_frame": "c"}
    body = "\n".join(json.dumps(row) for row in (header, *rows))
    path.write_text(body + "\n", encoding="utf-8")
    return path


def test_a_frame_that_produced_nothing_still_counts_as_processed(
        tmp_path: Path) -> None:
    """The distinction the whole resume rests on.

    A frame the detector found nothing in leaves no detection rows. Treating
    that as "not run" would re-run it forever; treating a missing frame as
    "run" would silently accept a truncated cache. The marker separates them,
    and the two sets differ exactly on the hardest frames.
    """
    path = written(tmp_path / "d.jsonl",
                   {"frame_id": "a", "label": "Car", "score": 0.9,
                    "box": [0, 0, 1, 1]},
                   {"kind": "frame", "frame_id": "a"},
                   {"kind": "frame", "frame_id": "b"})

    assert processed_frames(path) == {"a", "b"}

    _, detections = load_detections(path)
    assert detections["a"], "a produced a detection"
    assert detections["b"] == [], "b was run and produced nothing"
    assert "c" not in detections, "c was never run"


def test_progress_markers_are_not_loaded_as_detections(tmp_path: Path) -> None:
    """A marker counted as a detection would be a phantom object with no box,
    inflating the false positive count on exactly the frames that had none."""
    path = written(tmp_path / "d.jsonl", {"kind": "frame", "frame_id": "a"})

    _, detections = load_detections(path)

    assert detections == {"a": []}


def test_a_cache_that_was_never_started_reports_nothing_processed(
        tmp_path: Path) -> None:
    assert processed_frames(tmp_path / "absent.jsonl") == set()


def test_a_half_written_final_line_does_not_lose_the_frames_before_it(
        tmp_path: Path) -> None:
    """An interrupted run ends mid-line. Everything before that line is still
    good, and the frame it belonged to simply gets run again."""
    path = tmp_path / "d.jsonl"
    path.write_text(
        json.dumps({"kind": "header", "model": "m", "frames": 2,
                    "score_threshold": 0.05, "first_frame": "a",
                    "last_frame": "b"}) + "\n"
        + json.dumps({"kind": "frame", "frame_id": "a"}) + "\n"
        + '{"kind": "frame", "frame_i', encoding="utf-8")

    assert processed_frames(path) == {"a"}


# --- the letterbox, where coordinates silently go wrong ----------------------
@pytest.mark.parametrize("width,height", [(1242, 375), (640, 640), (100, 900)])
def test_the_letterbox_inverse_returns_the_original_point(
        width: int, height: int) -> None:
    """The transform that does not raise when it is wrong.

    An error here shifts every detection a few pixels, lowers IoU slightly, and
    reports a detector that is a little worse than it is. Round-tripping is the
    only way to catch it, because the output looks entirely reasonable.
    """
    box = Letterbox.fit(width, height, 640)

    for x, y in ((0.0, 0.0), (width / 2, height / 2), (float(width), float(height))):
        model_x = x * box.scale + box.pad_x
        model_y = y * box.scale + box.pad_y
        back_x, back_y = box.to_image(model_x, model_y)
        assert back_x == pytest.approx(x, abs=1e-6)
        assert back_y == pytest.approx(y, abs=1e-6)


def test_the_letterbox_preserves_aspect_ratio() -> None:
    """Squashing to the square instead of padding would distort every box in a
    way that looks like a mediocre detector."""
    box = Letterbox.fit(1242, 375, 640)

    assert box.scale == pytest.approx(640 / 1242)
    assert box.pad_x == pytest.approx(0.0)
    assert box.pad_y > 0


# --- non-maximum suppression -------------------------------------------------
def test_suppression_keeps_the_best_of_a_pile() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]],
                     dtype=np.float64)
    scores = np.array([0.5, 0.9, 0.3])

    assert _nms(boxes, scores, 0.5) == [1]


def test_suppression_keeps_genuinely_separate_objects() -> None:
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float64)

    assert sorted(_nms(boxes, np.array([0.9, 0.8]), 0.5)) == [0, 1]


def test_suppression_of_a_single_box_is_that_box() -> None:
    assert _nms(np.array([[0, 0, 10, 10]], dtype=np.float64),
                np.array([0.9]), 0.5) == [0]


def test_suppression_survives_a_zero_area_box() -> None:
    """A degenerate box makes the union zero. Without a guard that is a NaN
    that compares false against every threshold and silently survives."""
    boxes = np.array([[5, 5, 5, 5], [0, 0, 10, 10]], dtype=np.float64)

    assert len(_nms(boxes, np.array([0.9, 0.8]), 0.5)) == 2


# --- slice dimensions --------------------------------------------------------
def test_every_dimension_bins_every_object_it_claims_to() -> None:
    """A binner returning something outside its declared bins would produce a
    slice that silently contains nothing."""
    truth = [g for f in sorted(FIXTURES.glob("label_2/*.txt"))
             for g in load_labels(f)]
    assert truth

    for dim in DIMENSIONS:
        produced = {dim.of(item) for item in truth}
        produced.discard(None)
        unknown = produced - {str(b) for b in dim.bins}
        assert not unknown, f"{dim.name} produced bins it never declared: {unknown}"


def test_an_object_with_no_distance_is_outside_the_distance_dimension() -> None:
    """None, not a zero bin. An object with no 3D annotation has no distance,
    and putting it in "0-10 m" would be a lie rather than a gap."""
    assert dimension("distance").of(gt(distance_m=None)) is None
    assert dimension("box height").of(gt(distance_m=None)) is not None


@pytest.mark.parametrize("distance,expected", [
    (0.0, "0-10 m"), (9.99, "0-10 m"), (10.0, "10-20 m"),
    (49.9, "40-50 m"), (50.0, ">50 m"), (500.0, ">50 m"),
])
def test_distance_bands_are_half_open(distance: float, expected: str) -> None:
    """Boundaries decide which side an object lands on, so they are pinned."""
    assert dimension("distance").of(gt(distance_m=distance)) == expected


@pytest.mark.parametrize("occlusion,expected", [
    (0, "fully visible"), (1, "partly occluded"),
    (2, "largely occluded"), (3, "unknown"), (9, "unknown"),
])
def test_occlusion_levels_map_to_kittis_own_names(occlusion: int,
                                                  expected: str) -> None:
    assert dimension("occlusion").of(gt(occlusion=occlusion)) == expected


def test_asking_for_a_dimension_that_does_not_exist_says_what_does() -> None:
    with pytest.raises(KeyError, match="difficulty"):
        dimension("weather")


# --- the evaluation and the report -------------------------------------------
def small_evaluation():
    truth = {
        "a": [gt("Car", box=Box2D(0, 0, 100, 100), distance_m=5.0),
              gt("Car", box=Box2D(200, 0, 300, 100), distance_m=55.0)],
        "b": [gt("Pedestrian", box=Box2D(0, 0, 40, 120), distance_m=8.0)],
    }
    detections = {
        "a": [Detection("a", "Car", Box2D(0, 0, 100, 100), 0.9)],
        "b": [Detection("b", "Pedestrian", Box2D(0, 0, 40, 120), 0.8)],
    }
    return evaluate(detections, truth)


def test_the_evaluation_produces_a_cell_for_every_class_and_bin() -> None:
    result = small_evaluation()

    assert result.frames == 2
    assert result.objects == 3
    assert set(result.overall) == {"Car", "Pedestrian", "Cyclist"}
    for item in result.slices:
        assert {c.label for c in item.cells} == {"Car", "Pedestrian", "Cyclist"}


def test_a_class_absent_from_the_split_does_not_drag_the_headline_down() -> None:
    """Cyclist has no ground truth here, so its AP is undefined and must be
    skipped rather than averaged in as a zero."""
    result = small_evaluation()

    assert result.overall["Cyclist"].positives == 0
    assert result.headline == pytest.approx(
        (result.overall["Car"].average_precision
         + result.overall["Pedestrian"].average_precision) / 2)


def test_slices_carry_an_interval_where_they_carry_a_number() -> None:
    result = small_evaluation()

    for item in result.slices:
        for cell in item.cells:
            if cell.curve.positives:
                assert cell.interval is not None
                assert cell.interval.frames > 0
            else:
                assert cell.interval is None


def test_the_spread_names_where_the_best_and_worst_were() -> None:
    result = small_evaluation()
    best, worst, where_best, where_worst = result.spread("Car")

    if best == best:
        assert ":" in where_best and ":" in where_worst
        assert best >= worst


def test_the_report_is_one_self_contained_page() -> None:
    """No CDN, no external stylesheet, no build step. A report that needs
    infrastructure to read is a report nobody reads."""
    from ape.cache import Header

    html = render(small_evaluation(),
                  Header("m.onnx", 2, 0.05, "a", "b"))

    assert html.startswith("<!doctype html>")
    assert "<style>" in html
    assert "http://" not in html and "https://" not in html.split("github.com")[0]
    assert "cdn" not in html.lower()


def test_the_report_states_what_it_does_not_claim() -> None:
    """The limits are part of the artifact, not a footnote someone can drop."""
    from ape.cache import Header

    html = render(small_evaluation(), Header("m.onnx", 2, 0.05, "a", "b"))

    for phrase in ("Cyclist", "SOTIF", "pycocotools", "confidence interval"):
        assert phrase in html, f"the report no longer mentions {phrase}"


def test_the_report_survives_a_class_with_no_objects() -> None:
    from ape.cache import Header

    html = render(evaluate({}, {"a": [gt("Car")]}),
                  Header("m.onnx", 1, 0.05, "a", "a"))

    assert "n/a" in html or "-" in html


# --- difficulty, which the report leads with ---------------------------------
@pytest.mark.parametrize("tier", ["easy", "moderate", "hard"])
def test_every_difficulty_tier_is_reported(tier: str) -> None:
    result = small_evaluation()

    assert tier in result.by_difficulty
    assert set(result.by_difficulty[tier]) == {"Car", "Pedestrian", "Cyclist"}


def test_harder_tiers_include_at_least_as_many_objects() -> None:
    """Cumulative, as KITTI defines it. A hard tier containing fewer objects
    than the easy one would mean the tiers are being read as exclusive."""
    result = small_evaluation()

    easy = result.by_difficulty["easy"]["Car"].positives
    hard = result.by_difficulty["hard"]["Car"].positives

    assert hard >= easy


def test_the_ignored_tier_is_a_slice_and_not_a_difficulty() -> None:
    """KITTI excludes these from its benchmark, so they are worth looking at
    as a slice but must not appear as a scored tier."""
    result = small_evaluation()

    assert Difficulty.IGNORED.value not in result.by_difficulty
    assert any(s.bin == Difficulty.IGNORED.value for s in result.slices)
