"""Price the accuracy against the latency, on the same frames.

    uv run --extra infer python scripts/compare_models.py

WHY THIS EXISTS. M2 set a budget of 1500 frames in five minutes and missed it,
at 799 seconds, and the profiling showed the cost is the model's forward pass
rather than the pipeline around it. That leaves an obvious question the project
had been ducking: a smaller model would fit the budget, so what does it cost?

"YOLOv8n would fit at a cost in accuracy" is the kind of sentence that sounds
like analysis and contains no information. This measures the cost.

BOTH MODELS SEE THE SAME FRAMES, in the same order, through the same
preprocessing, scored by the same code. That is the only way the difference is
attributable to the model. Latency is measured separately from the cached run,
on a warm session, because a first call includes graph optimisation that no
steady-state deployment pays.

WHAT IT DELIBERATELY DOES NOT DO: pick one. Which model to ship depends on the
target hardware, the frame rate the function needs and how much of the accuracy
matters at the operating point, and only the last of those is in this
repository.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.cache import load_detections  # noqa: E402
from ape.classes import EVALUATED, HEADLINE, neutral_labels  # noqa: E402
from ape.detect import Detector  # noqa: E402
from ape.evaluate import IOU  # noqa: E402
from ape.kitti import frame_ids, load_labels  # noqa: E402
from ape.localisation import diagnose  # noqa: E402
from ape.match import assign  # noqa: E402
from ape.metrics import average_precision  # noqa: E402
from ape.operating import best_recall, sweep  # noqa: E402

#: The M2 criterion, kept as a number so the verdict is computed rather than
#: asserted: 1500 frames in five minutes.
BUDGET_FRAMES, BUDGET_SECONDS = 1500, 300


def latency(model: Path, images: list[Path], threads: int,
            warmup: int = 3) -> tuple[float, float]:
    """Median and 95th percentile seconds per frame, on a warm session.

    Median rather than mean because a single scheduling hiccup on a laptop
    moves a mean and says nothing about the model. The 95th is reported too,
    since a perception function misses its slot on the slow frames, not the
    typical ones.
    """
    detector = Detector(model, score_threshold=0.05, threads=threads)
    for path in images[:warmup]:
        detector.detect(path, path.stem)

    taken = []
    for path in images[warmup:]:
        started = time.perf_counter()
        detector.detect(path, path.stem)
        taken.append(time.perf_counter() - started)
    ordered = sorted(taken)
    return statistics.median(taken), ordered[int(len(ordered) * 0.95)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolov8s"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/models.json")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--latency-frames", type=int, default=40)
    args = parser.parse_args()

    images = [args.data / "image_2" / f"{f}.png"
              for f in frame_ids(args.data / "label_2")[:args.latency_frames]]

    rows = []
    for name in args.models:
        cache = ROOT / f"outputs/detections-{name}.jsonl"
        if not cache.exists():
            raise SystemExit(
                f"{cache} is missing. Run scripts/run_inference.py --model "
                f"models/{name}.onnx --out {cache.name} first; comparing a "
                f"model against a run it did not produce would be meaningless.")

        header, detections = load_detections(cache)
        split = frame_ids(args.data / "label_2")[:header.frames]
        truth = {f: load_labels(args.data / "label_2" / f"{f}.txt") for f in split}

        print(f"{name}: measuring latency over {len(images)} frames", flush=True)
        median, p95 = latency(ROOT / f"models/{name}.onnx", images, args.threads)

        row: dict = {
            "model": name,
            "size_mb": (ROOT / f"models/{name}.onnx").stat().st_size / 1e6,
            "frames": header.frames,
            "median_ms": median * 1000,
            "p95_ms": p95 * 1000,
            "fps": 1 / median,
            "budget_seconds": BUDGET_FRAMES * median,
            "meets_budget": BUDGET_FRAMES * median <= BUDGET_SECONDS,
            "ap": {}, "ceiling": {}, "mislocation_share": {},
        }
        for label in EVALUATED:
            neutral = neutral_labels(label)
            whole = assign(detections, truth, label, neutral, IOU)
            row["ap"][label] = average_precision(whole).average_precision
            row["ceiling"][label] = best_recall(sweep(whole, len(truth)))
            row["mislocation_share"][label] = diagnose(
                detections, truth, label, neutral, tight=IOU).mislocation_share
        row["headline"] = statistics.fmean(row["ap"][c] for c in HEADLINE)
        rows.append(row)

    print(f"\n{'model':<10} {'MB':>6} {'median':>9} {'p95':>9} {'fps':>6} "
          f"{'1500 frames':>12} {'budget':>8}")
    for row in rows:
        verdict = "MET" if row["meets_budget"] else "MISSED"
        print(f"{row['model']:<10} {row['size_mb']:>6.1f} "
              f"{row['median_ms']:>8.0f}ms {row['p95_ms']:>8.0f}ms "
              f"{row['fps']:>6.1f} {row['budget_seconds']:>11.0f}s {verdict:>8}")

    print(f"\n{'model':<10} " + "".join(f"{c:>12}" for c in EVALUATED)
          + f"{'headline':>10}")
    for row in rows:
        print(f"{row['model']:<10} "
              + "".join(f"{row['ap'][c]:>12.3f}" for c in EVALUATED)
              + f"{row['headline']:>10.3f}")

    if len(rows) == 2:
        small, large = sorted(rows, key=lambda r: r["size_mb"])
        speedup = large["median_ms"] / small["median_ms"]
        print(f"\n{small['model']} is {speedup:.2f}x faster than "
              f"{large['model']}, and costs:")
        for label in HEADLINE:
            lost = large["ap"][label] - small["ap"][label]
            share = lost / large["ap"][label] if large["ap"][label] else float("nan")
            print(f"  {label:<11} {lost:+.3f} AP ({share:+.1%})")
        for label in HEADLINE:
            print(f"  {label:<11} recall ceiling "
                  f"{large['ceiling'][label]:.1%} -> {small['ceiling'][label]:.1%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"budget_frames": BUDGET_FRAMES, "budget_seconds": BUDGET_SECONDS,
         "models": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
