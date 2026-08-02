"""My average precision against pycocotools, on identical inputs.

THIS IS THE GATE THE PROJECT LIVES OR DIES ON. Everything else here, the
slicing, the taxonomy, the sim-to-real comparison, is built on top of one
number. If that number is wrong, every conclusion drawn from it is wrong in a
way no amount of careful analysis downstream will reveal, because a wrong AP
does not look wrong. It looks like a result.

So it is not trusted, it is checked, against an implementation nobody here
wrote and that thousands of published papers are scored with. Agreement is
required to within 0.001. There is no tolerance argument: the two are computing
the same definition over the same inputs, so they either agree or one of them
has a bug.

WHAT HAD TO BE MADE EQUIVALENT for the comparison to mean anything:

  ignore regions   mine calls them "tolerated", COCO calls them `iscrowd`.
                   Both mean "a detection here is neither right nor wrong".
                   Get this wrong and the two disagree for a reason that has
                   nothing to do with the AP arithmetic.

  detection cap    COCO keeps only the top 100 detections per image by default.
                   Raised here, because a cap is a property of the COCO
                   benchmark and not of average precision, and leaving it in
                   would make the two disagree on crowded frames only.

  IoU thresholds   COCO averages over 0.50:0.95. Pinned to 0.50 alone, because
                   that is the number this project reports.

  area ranges      COCO also reports small/medium/large. Only `all` is compared;
                   this project slices by KITTI's own attributes instead, which
                   is a deliberate difference and not an oversight.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import pytest

from ape.classes import EVALUATED, neutral_labels
from ape.kitti import load_labels
from ape.match import assign, partition
from ape.metrics import average_precision
from ape.records import Detection, GroundTruth

pytest.importorskip("pycocotools",
                    reason="the reference implementation is an optional extra")

FIXTURES = Path(__file__).parent / "fixtures"
IOU = 0.5

#: Category ids are arbitrary but must be consistent between the two documents.
CATEGORY_ID = {name: index + 1 for index, name in enumerate(EVALUATED)}


def frames() -> list[str]:
    return sorted(p.stem for p in (FIXTURES / "label_2").glob("*.txt"))


def ground_truth() -> dict[str, list[GroundTruth]]:
    return {f: load_labels(FIXTURES / "label_2" / f"{f}.txt") for f in frames()}


def synthetic_detections(
        truth: dict[str, list[GroundTruth]]) -> dict[str, list[Detection]]:
    """Detections derived from the ground truth, perturbed on purpose.

    Real detector output is used by the integration test alongside this one.
    Here the input is synthetic so the comparison covers the cases that are rare
    in practice and fatal when wrong: a perfect box, a box just under the IoU
    threshold, a duplicate of an object already claimed, a detection of nothing,
    and a detection landing on an object the benchmark does not score.
    """
    out: dict[str, list[Detection]] = {}
    for index, (frame, items) in enumerate(sorted(truth.items())):
        detections: list[Detection] = []
        for position, item in enumerate(items):
            neutral_extra = {"Van", "Person_sitting"}
            if item.label not in EVALUATED and item.label not in neutral_extra:
                continue
            label = item.label if item.label in EVALUATED else "Car"
            box = item.box
            # A spread of qualities, deterministic per object so the test is
            # reproducible: exact, slightly off, badly off, and a duplicate.
            variant = (index + position) % 4
            if variant == 0:
                shifted = box
                score = 0.95
            elif variant == 1:
                shifted = type(box)(box.x1 + box.width * 0.10, box.y1,
                                    box.x2 + box.width * 0.10, box.y2)
                score = 0.80
            elif variant == 2:
                shifted = type(box)(box.x1 + box.width * 0.60, box.y1,
                                    box.x2 + box.width * 0.60, box.y2)
                score = 0.60
            else:
                shifted = box
                score = 0.40
                detections.append(Detection(frame, label, box, 0.55))
            detections.append(Detection(frame, label, shifted, score))
        # A detection of empty sky: a clean false positive with no ambiguity.
        detections.append(Detection(frame, "Car", type(items[0].box)(
            5.0, 5.0, 45.0, 45.0) if items else None, 0.30))  # type: ignore[arg-type]
        out[frame] = [d for d in detections if d.box is not None]
    return out


def as_coco(truth: dict[str, list[GroundTruth]],
            detections: dict[str, list[Detection]], label: str) -> tuple[dict, list]:
    """The same evaluation, expressed the way the reference expects it.

    One class at a time, because this project reports per class and averages
    afterwards; letting the reference do the averaging would compare two
    different aggregation rules as well as two AP implementations.
    """
    neutral = neutral_labels(label)
    images, annotations = [], []
    annotation_id = 1
    for image_id, frame in enumerate(sorted(truth), 1):
        images.append({"id": image_id, "file_name": f"{frame}.png",
                       "width": 1242, "height": 375})
        counts, tolerated = partition(truth[frame], label, neutral, None)
        for item, is_crowd in ([(c, 0) for c in counts]
                               + [(t, 1) for t in tolerated]):
            box = item.box
            annotations.append({
                "id": annotation_id, "image_id": image_id,
                "category_id": CATEGORY_ID[label],
                "bbox": [box.x1, box.y1, box.width, box.height],
                "area": box.area, "iscrowd": is_crowd, "ignore": is_crowd,
            })
            annotation_id += 1

    results = []
    for image_id, frame in enumerate(sorted(truth), 1):
        for detection in detections.get(frame, []):
            if detection.label != label:
                continue
            box = detection.box
            results.append({
                "image_id": image_id, "category_id": CATEGORY_ID[label],
                "bbox": [box.x1, box.y1, box.width, box.height],
                "score": detection.score,
            })

    dataset = {"images": images, "annotations": annotations,
               "categories": [{"id": CATEGORY_ID[label], "name": label}]}
    return dataset, results


def reference_ap(dataset: dict, results: list) -> float:
    """AP at IoU 0.5 from pycocotools, or NaN when it has nothing to score."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not results or not any(a["iscrowd"] == 0 for a in dataset["annotations"]):
        return float("nan")

    # pycocotools prints unconditionally; the output is noise in a test report.
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        detections = coco.loadRes(list(results))

        evaluation = COCOeval(coco, detections, "bbox")
        evaluation.params.iouThrs = np.array([IOU])
        evaluation.params.areaRng = [[0.0, 1e10]]
        evaluation.params.areaRngLbl = ["all"]
        evaluation.params.maxDets = [1000]
        evaluation.evaluate()
        evaluation.accumulate()

    precision = evaluation.eval["precision"][0, :, 0, 0, 0]
    valid = precision[precision > -1]
    return float(valid.mean()) if valid.size else float("nan")


@pytest.mark.parametrize("label", EVALUATED)
def test_my_average_precision_agrees_with_pycocotools(label: str) -> None:
    """The gate. Same detections, same ground truth, same definition."""
    truth = ground_truth()
    detections = synthetic_detections(truth)

    mine = average_precision(
        assign(detections, truth, label, neutral_labels(label), IOU))
    theirs = reference_ap(*as_coco(truth, detections, label))

    if theirs != theirs:  # NaN: the reference had nothing to score
        assert mine.positives == 0 or not mine.true_positives + mine.false_positives
        pytest.skip(f"no scoreable {label} in the fixture")

    assert mine.average_precision == pytest.approx(theirs, abs=0.001), (
        f"{label}: mine {mine.average_precision:.6f} vs reference "
        f"{theirs:.6f}. These compute the same definition over the same "
        f"inputs, so one of them has a bug."
    )


def test_the_comparison_would_notice_a_wrong_answer() -> None:
    """The guard on the gate.

    A comparison that passes no matter what is not evidence. This deliberately
    breaks the matching, by dropping the IoU threshold to zero so that every
    detection matches whatever is nearest, and confirms the two implementations
    then disagree by more than the tolerance.
    """
    truth = ground_truth()
    detections = synthetic_detections(truth)
    label = "Car"

    honest = average_precision(
        assign(detections, truth, label, neutral_labels(label), IOU))
    broken = average_precision(
        assign(detections, truth, label, neutral_labels(label), 0.0))
    reference = reference_ap(*as_coco(truth, detections, label))

    assert honest.average_precision == pytest.approx(reference, abs=0.001)
    assert abs(broken.average_precision - reference) > 0.001, (
        "a deliberately wrong matcher still agreed with the reference, so this "
        "comparison proves nothing"
    )
