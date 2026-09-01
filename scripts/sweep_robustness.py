"""Score the detector under each perturbation, and write the degradation curves.

    uv run --extra infer --extra report python scripts/sweep_robustness.py --frames 500

Runs inference once per (perturbation, strength) over the same frames, so every
point on a curve differs from the baseline in exactly one stated way and the
ground truth is identical throughout. Writes:

    outputs/robustness.csv     one row per perturbation, strength and class
    outputs/robustness.md      the same as a table, for the README

The baseline is the strength-0 run of each perturbation rather than a separate
unperturbed pass. That is deliberate: if a perturbation's identity case ever
disagrees with the others, the harness is wrong and the sweep says so instead
of quietly comparing against a different pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.classes import EVALUATED, neutral_labels  # noqa: E402
from ape.evaluate import IOU  # noqa: E402
from ape.kitti import frame_ids, load_labels  # noqa: E402
from ape.match import assign  # noqa: E402
from ape.metrics import average_precision, false_negative_rate  # noqa: E402
from ape.perturb import PERTURBATIONS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8s.onnx")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--csv", type=Path, default=ROOT / "outputs/robustness.csv")
    parser.add_argument("--md", type=Path, default=ROOT / "outputs/robustness.md")
    args = parser.parse_args()

    from PIL import Image

    from ape.detect import Detector
    from ape.perturb import apply as perturb

    split = frame_ids(args.data / "label_2")[: args.frames]
    truth = {f: load_labels(args.data / "label_2" / f"{f}.txt") for f in split}
    detector = Detector(args.model, score_threshold=args.score, threads=args.threads)

    total = sum(len(p.strengths) for p in PERTURBATIONS)
    print(f"{len(split)} frames x {total} configurations")

    rows: list[dict[str, object]] = []
    done = 0
    for perturbation in PERTURBATIONS:
        for strength in perturbation.strengths:
            started = time.time()
            detections: dict[str, list] = {}
            for frame in split:
                with Image.open(args.data / "image_2" / f"{frame}.png") as handle:
                    image = handle.convert("RGB")
                    if strength != 0:
                        image = perturb(perturbation.name, image, strength)
                    detections[frame] = detector.detect_image(image, frame)

            for label in EVALUATED:
                curve = average_precision(
                    assign(detections, truth, label, neutral_labels(label), IOU))
                rows.append({
                    "perturbation": perturbation.name,
                    "unit": perturbation.unit,
                    "strength": strength,
                    "label": label,
                    "ap": round(curve.average_precision, 4),
                    "max_recall": round(curve.best_recall, 4),
                    "fnr": round(false_negative_rate(curve), 4),
                    "positives": curve.positives,
                })
            done += 1
            print(f"  [{done}/{total}] {perturbation.name} {strength:+g} "
                  f"in {time.time() - started:.0f}s")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    args.md.write_text(_markdown(rows, len(split)), encoding="utf-8")
    print(f"wrote {args.csv.relative_to(ROOT)}\nwrote {args.md.relative_to(ROOT)}")
    return 0


def _markdown(rows: list[dict[str, object]], frames: int) -> str:
    out = ["# Metamorphic robustness", "",
           f"AP at IoU {IOU}, {frames} KITTI frames, same ground truth throughout. "
           "Strength 0 is the unperturbed baseline for that perturbation.", ""]
    for perturbation in PERTURBATIONS:
        mine = [r for r in rows if r["perturbation"] == perturbation.name]
        if not mine:
            continue
        out += [f"## {perturbation.name} ({perturbation.unit})", "",
                "| strength | " + " | ".join(f"{c} AP" for c in EVALUATED) + " |",
                "|---" * (len(EVALUATED) + 1) + "|"]
        for strength in perturbation.strengths:
            cells = []
            for label in EVALUATED:
                hit = next((r for r in mine if r["strength"] == strength
                            and r["label"] == label), None)
                cells.append(f"{hit['ap']:.3f}" if hit else "-")
            out.append(f"| {strength:+g} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.exit(main())
