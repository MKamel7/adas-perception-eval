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

from ape.cache import processed_frames  # noqa: E402
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
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="start over rather than continuing an existing run")
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

    # RESUMABLE, because a full-split run takes an hour and anything can end it.
    # The first version rewrote from scratch, so an interruption at 82% left a
    # file whose header claimed 7481 frames and whose contents covered 6178: a
    # cache lying about its own extent, which is the one thing a cache must
    # never do.
    done = processed_frames(args.out) if args.resume else set()
    remaining = [f for f in frames if f not in done]
    if done:
        print(f"resuming: {len(done)} frames already done, "
              f"{len(remaining)} to go", flush=True)
    if not remaining:
        print(f"all {len(frames)} frames already complete -> {args.out}")
        return 0

    started = time.time()
    written = 0
    mode = "a" if done else "w"
    with args.out.open(mode, encoding="utf-8") as handle:
        if mode == "w":
            handle.write(json.dumps({
                "kind": "header", "model": args.model.name, "frames": len(frames),
                "score_threshold": args.score, "threads": args.threads,
                "first_frame": frames[0], "last_frame": frames[-1],
            }) + "\n")
        for index, frame in enumerate(remaining, 1):
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
            # One marker per frame, so a frame the detector found nothing in is
            # distinguishable from one that was never run. Without it a resume
            # cannot tell those apart, and neither can the evaluation.
            handle.write(json.dumps({"kind": "frame", "frame_id": frame}) + "\n")
            handle.flush()
            if index % 100 == 0:
                rate = index / (time.time() - started)
                print(f"  {index}/{len(remaining)} frames, {rate:.1f}/s", flush=True)

    elapsed = time.time() - started
    total = len(processed_frames(args.out))
    print(f"\n{len(remaining)} frames in {elapsed:.0f}s "
          f"({len(remaining) / elapsed:.1f}/s), {written} detections this pass")
    print(f"{total}/{len(frames)} frames complete -> {args.out}")
    if total != len(frames):
        print("INCOMPLETE. Run again to resume: the cache records exactly which "
              "frames are done, so nothing is repeated and nothing is skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
