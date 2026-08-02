"""Render example frames for the triggering conditions.

    uv run --extra infer python scripts/render_examples.py

A number in a table says a slice is bad. An image says what bad looks like, and
for a perception argument that is not decoration: "pedestrian AP 0.007 beyond
30 m" and a picture of a person standing in the road with no box around them are
the same fact, and only one of them survives being read quickly.

WHAT IS DRAWN AND WHY IT IS NOT CHERRY-PICKED. The frames are chosen by a rule,
not by eye: for each condition, the frame containing the most ground-truth
objects that fall in that slice AND were missed. Choosing them by hand would
make them illustrations of an argument rather than evidence for it, which is
exactly the failure mode the slicing scheme exists to avoid.

Green is ground truth. Red is a ground-truth object that no detection matched.
Blue is a detection. A red box with no blue near it is the failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.cache import load_detections  # noqa: E402
from ape.classes import neutral_labels  # noqa: E402
from ape.evaluate import IOU  # noqa: E402
from ape.kitti import frame_ids, load_labels  # noqa: E402
from ape.match import partition  # noqa: E402
from ape.slices import dimension  # noqa: E402

#: condition -> (class, slice dimension, bins that condition covers)
EXAMPLES = {
    "TC-01": ("Pedestrian", "distance", {"30-40 m", "40-50 m", ">50 m"}),
    "TC-02": ("Pedestrian", "occlusion", {"largely occluded", "partly occluded"}),
    "TC-03": ("Pedestrian", "box height", {"0-25 px", "25-40 px"}),
    "TC-04": ("Car", "distance", {">50 m"}),
    "TC-05": ("Pedestrian", "truncation", {"30-50%", ">50%"}),
}

GREEN, RED, BLUE = (60, 200, 120), (220, 60, 80), (70, 140, 240)


def missed(truth, detections, label):
    """Ground truth of this class that no detection claimed, greedily by score."""
    counts, _ = partition(truth, label, neutral_labels(label), None)
    claimed: set[int] = set()
    for detection in sorted(detections, key=lambda d: d.score, reverse=True):
        if detection.label != label:
            continue
        best, index = 0.0, -1
        for i, candidate in enumerate(counts):
            if i in claimed:
                continue
            overlap = detection.box.iou(candidate.box)
            if overlap > best:
                best, index = overlap, i
        if index >= 0 and best >= IOU:
            claimed.add(index)
    return [item for i, item in enumerate(counts) if i not in claimed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path,
                        default=ROOT / "outputs/detections-yolov8s.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/examples")
    parser.add_argument("--score", type=float, default=0.25,
                        help="confidence to draw at; the evaluation keeps a much "
                             "lower tail, which would clutter the picture")
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    header, detections = load_detections(args.detections)
    split = frame_ids(args.data / "label_2")[:header.frames]
    args.out.mkdir(parents=True, exist_ok=True)

    for condition, (label, dim_name, bins) in EXAMPLES.items():
        binner = dimension(dim_name).of

        best_frame, best_missed = None, []
        for frame in split:
            truth = load_labels(args.data / "label_2" / f"{frame}.txt")
            in_slice = [g for g in truth if binner(g) in bins and g.label == label]
            if not in_slice:
                continue
            gone = [g for g in missed(truth, detections.get(frame, []), label)
                    if binner(g) in bins]
            if len(gone) > len(best_missed):
                best_frame, best_missed = frame, gone

        if best_frame is None:
            print(f"{condition}: no frame contains a missed {label} in {dim_name}")
            continue

        truth = load_labels(args.data / "label_2" / f"{best_frame}.txt")
        found = [d for d in detections.get(best_frame, []) if d.score >= args.score]

        with Image.open(args.data / "image_2" / f"{best_frame}.png") as handle:
            canvas = handle.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for item in truth:
            if item.label in {"DontCare"}:
                continue
            box = item.box
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=GREEN, width=2)
        for detection in found:
            box = detection.box
            draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=BLUE, width=2)
        for item in best_missed:
            box = item.box
            draw.rectangle([box.x1 - 2, box.y1 - 2, box.x2 + 2, box.y2 + 2],
                           outline=RED, width=3)

        target = args.out / f"{condition}-{best_frame}.png"
        canvas.save(target)
        print(f"{condition}: frame {best_frame}, {len(best_missed)} missed "
              f"{label} in {dim_name} -> {target.name}")

    print(f"\ngreen = ground truth, blue = detection above {args.score}, "
          f"red = a {'/'.join({v[0] for v in EXAMPLES.values()})} nothing found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
