"""Render the demo scene: watch pedestrian detection fail with distance.

    uv run --extra infer python scripts/render_demo.py

WHAT IT SHOWS, and why this subject and not a montage of the pipeline running.
The project's finding is that pedestrian AP falls from 0.689 under 10 m to 0.000
beyond 50 m. That is a table. This walks a viewer through the same fact by
ordering real frames by the distance of the pedestrians in them, so the boxes
visibly stop appearing while the caption states the measured AP for the band
being shown.

Nothing here is staged. The frames are selected by the same rule the example
stills use, the boxes are the real detector output at the same threshold, and
the AP figures are read from outputs/results.json rather than typed in.

Green is ground truth. Blue is a detection. Red is a ground-truth pedestrian
that no detection matched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

GREEN, RED, BLUE, INK, PAPER = ((60, 200, 120), (235, 60, 80), (70, 150, 245),
                                (245, 246, 248), (18, 20, 24))

#: Bands in the order the scene walks through them, near to far.
BANDS = ("0-10 m", "10-20 m", "20-30 m", "30-40 m", "40-50 m", ">50 m")

#: Frames held per band, and how long each is held. 6 bands x 5 frames x 0.7 s
#: is about 21 seconds, which is the length the milestone asked for.
PER_BAND, HOLD = 5, 0.7


def font(size: int):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def missed(truth, detections, label):
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
    return [g for i, g in enumerate(counts) if i not in claimed]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path,
                        default=ROOT / "outputs/detections.jsonl")
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--results", type=Path, default=ROOT / "outputs/results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/p3-demo.mp4")
    parser.add_argument("--score", type=float, default=0.25)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    from PIL import Image, ImageDraw

    header, detections = load_detections(args.detections)
    split = frame_ids(args.data / "label_2")[:header.frames]
    binner = dimension("distance").of

    results = json.loads(args.results.read_text(encoding="utf-8"))
    ap = {}
    for item in results["slices"]:
        if item["dimension"] == "distance":
            for cell in item["cells"]:
                if cell["label"] == "Pedestrian":
                    ap[item["bin"]] = cell["ap"]

    #: band -> frames containing the most pedestrians in that band, richest first
    candidates: dict[str, list[tuple[int, str]]] = {b: [] for b in BANDS}
    for frame in split:
        truth = load_labels(args.data / "label_2" / f"{frame}.txt")
        for band in BANDS:
            n = sum(1 for g in truth
                    if g.label == "Pedestrian" and binner(g) == band)
            if n:
                candidates[band].append((n, frame))
    for band in BANDS:
        candidates[band].sort(reverse=True)

    staging = ROOT / "outputs/.demo_frames"
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob("*.png"):
        old.unlink()

    big, small = font(30), font(21)
    index = 0
    for band in BANDS:
        chosen = [f for _, f in candidates[band][:PER_BAND]]
        if not chosen:
            print(f"  {band}: no frames, skipped")
            continue
        for frame in chosen:
            truth = load_labels(args.data / "label_2" / f"{frame}.txt")
            found = [d for d in detections.get(frame, [])
                     if d.score >= args.score and d.label == "Pedestrian"]
            gone = [g for g in missed(truth, detections.get(frame, []), "Pedestrian")
                    if binner(g) == band]

            with Image.open(args.data / "image_2" / f"{frame}.png") as handle:
                photo = handle.convert("RGB")

            # Both dimensions must be EVEN or libx264 with yuv420p refuses the
            # whole encode: chroma is subsampled 2x1 and an odd height has no
            # valid plane size. 375 + 96 = 471 fails; the image is padded to 376
            # rather than the bar being made 97, so the photo keeps a whole
            # number of pixels per source row.
            canvas = Image.new("RGB", (1242, 376 + 96), PAPER)
            canvas.paste(photo.resize((1242, 376)), (0, 0))
            draw = ImageDraw.Draw(canvas)

            for item in truth:
                if item.label != "Pedestrian":
                    continue
                box = item.box
                draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                               outline=GREEN, width=2)
            for detection in found:
                box = detection.box
                draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                               outline=BLUE, width=2)
            for item in gone:
                box = item.box
                draw.rectangle([box.x1 - 3, box.y1 - 3, box.x2 + 3, box.y2 + 3],
                               outline=RED, width=3)

            score = ap.get(band)
            draw.text((22, 386), f"Pedestrians at {band}", font=big, fill=INK)
            draw.text((22, 425),
                      f"measured AP {score:.3f}" if score is not None else "",
                      font=small, fill=RED if (score or 0) < 0.1 else INK)
            draw.text((430, 425),
                      "green: annotated    blue: detected    red: annotated, "
                      "nothing found", font=small, fill=(150, 156, 166))
            draw.text((980, 386), f"frame {frame}", font=small,
                      fill=(150, 156, 166))

            for _ in range(round(HOLD * args.fps)):
                canvas.save(staging / f"{index:05d}.png")
                index += 1
        print(f"  {band}: {len(chosen)} frames, AP {ap.get(band, float('nan')):.3f}")

    ffmpeg = next(iter(Path(
        "C:/Users/mkame/AppData/Local/Microsoft/WinGet/Packages"
    ).glob("**/ffmpeg.exe")), None)
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; the frames are in " + str(staging))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ffmpeg), "-y", "-v", "error", "-framerate", str(args.fps),
                    "-i", str(staging / "%05d.png"), "-c:v", "libx264",
                    "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.out)], check=True)
    for old in staging.glob("*.png"):
        old.unlink()
    staging.rmdir()

    print(f"\nwrote {args.out} ({index / args.fps:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
