"""Run the detector over a split and cache the detections.

    uv run --extra infer python scripts/run_inference.py --limit 1500

Detections are cached to JSONL so that everything downstream, matching, metrics,
slicing and the report, can be re-run in seconds without paying for inference
again. That is not just convenience: it means a change to the METRIC can be
compared against the same detections, so a difference in the result is
attributable to the change rather than to a different run of the model.

The split is written into the cache header. An evaluation whose frame list is
not recorded cannot be reproduced or compared with another.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.detect import Detector  # noqa: E402
from ape.kitti import frame_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8s.onnx")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/detections.jsonl")
    parser.add_argument("--limit", type=int, default=1500,
                        help="frames to run, from the start of the sorted split")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--score", type=float, default=0.05,
                        help="keep low-confidence boxes: average precision is "
                             "computed over the whole precision-recall curve, "
                             "and cutting at 0.25 truncates it")
    args = parser.parse_args()

    frames = frame_ids(args.data / "label_2")[:args.limit]
    if not frames:
        raise SystemExit(f"no frames under {args.data / 'label_2'}")

    detector = Detector(args.model, score_threshold=args.score, threads=args.threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "kind": "header", "model": args.model.name, "frames": len(frames),
            "score_threshold": args.score, "threads": args.threads,
            "first_frame": frames[0], "last_frame": frames[-1],
        }) + "\n")
        for index, frame in enumerate(frames, 1):
            image = args.data / "image_2" / f"{frame}.png"
            if not image.exists():
                raise SystemExit(
                    f"{image} is missing. Run scripts/fetch_kitti.py --images; "
                    f"evaluating the frames that happen to be present would "
                    f"report a number for a split nobody chose.")
            for detection in detector.detect(image, frame):
                handle.write(json.dumps({
                    "frame_id": detection.frame_id, "label": detection.label,
                    "score": round(detection.score, 5),
                    "box": [round(v, 2) for v in (detection.box.x1, detection.box.y1,
                                                  detection.box.x2, detection.box.y2)],
                }) + "\n")
                written += 1
            if index % 100 == 0:
                rate = index / (time.time() - started)
                print(f"  {index}/{len(frames)} frames, {rate:.1f}/s", flush=True)

    elapsed = time.time() - started
    print(f"\n{len(frames)} frames in {elapsed:.0f}s "
          f"({len(frames)/elapsed:.1f}/s), {written} detections -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
