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

Green is a pedestrian in the captioned band that was found. Red is one that was
not, annotated with the best overlap any detection achieved so a viewer can see
WHY it counts as a miss. Blue is a detection. Grey is a pedestrian outside the
captioned band, drawn faintly because it is in the frame but is not what this
section is about.

THREE THINGS THE FIRST VERSION GOT WRONG, all of which made the picture argue
something different from the caption:

  it drew every pedestrian in green regardless of band, so a frame captioned
  "0-10 m" showed objects at 20-30 m as though they were part of the claim

  it matched against every detection down to 0.05 confidence while drawing only
  those above 0.25, so the verdict and the image rested on different evidence
  and a box could be marked missed with a visible detection sitting on it

  it never showed the overlap, so a near miss at IoU 0.41 and a total absence
  looked identical

The picture is now drawn at a single stated operating point and the matching
uses exactly the detections that are visible. The AP in the caption is a
different quantity, computed over the whole precision-recall curve, and the
caption says so rather than letting the two be confused.
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

GREEN, RED, BLUE, GREY, INK, PAPER = ((60, 200, 120), (235, 60, 80),
                                      (70, 150, 245), (120, 126, 136),
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


def match(truth, detections, label):
    """Which pedestrians were found, and the best overlap on each that was not.

    Returns (found, missed) where missed carries the best IoU any visible
    detection achieved on it. That number is the difference between "the
    detector did not see this" and "the detector saw it and boxed it badly",
    which are different failures with different fixes and looked identical in
    the first version of this scene.
    """
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

    found = [g for i, g in enumerate(counts) if i in claimed]
    gone = [(g, max((d.box.iou(g.box) for d in detections
                     if d.label == label), default=0.0))
            for i, g in enumerate(counts) if i not in claimed]
    return found, gone


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path,
                        default=ROOT / "outputs/detections-yolov8s.jsonl")
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
    candidates: dict[str, list[tuple[float, int, str]]] = {b: [] for b in BANDS}
    for frame in split:
        truth = load_labels(args.data / "label_2" / f"{frame}.txt")
        for band in BANDS:
            peds = [g for g in truth if g.label == "Pedestrian"]
            inside = sum(1 for g in peds if binner(g) == band)
            outside = len(peds) - inside
            if inside:
                # Rank by how much of the frame is actually about this band.
                # Ranking on the raw count picked busy frames full of
                # pedestrians at every other distance, which is how a frame
                # captioned "0-10 m" ended up showing objects at 20-30 m.
                candidates[band].append((inside - 0.5 * outside, inside, frame))
    for band in BANDS:
        candidates[band].sort(reverse=True)

    staging = ROOT / "outputs/.demo_frames"
    staging.mkdir(parents=True, exist_ok=True)
    for old in staging.glob("*.png"):
        old.unlink()

    big, small, tiny = font(30), font(21), font(15)
    index = 0
    for band in BANDS:
        chosen = [f for _, _, f in candidates[band][:PER_BAND]]
        if not chosen:
            print(f"  {band}: no frames, skipped")
            continue
        for frame in chosen:
            truth = load_labels(args.data / "label_2" / f"{frame}.txt")
            # ONE set of detections, used both for drawing and for deciding.
            # Matching against everything down to 0.05 while drawing only what
            # cleared 0.25 meant the verdict and the image rested on different
            # evidence, so a box could be marked missed with a visible
            # detection sitting on it.
            visible = [d for d in detections.get(frame, [])
                       if d.score >= args.score and d.label == "Pedestrian"]
            found, gone = match(truth, visible, "Pedestrian")

            with Image.open(args.data / "image_2" / f"{frame}.png") as handle:
                photo = handle.convert("RGB")

            # Both dimensions must be EVEN or libx264 with yuv420p refuses the
            # whole encode: chroma is subsampled 2x1 and an odd height has no
            # valid plane size. 375 + 96 = 471 fails; the image is padded to 376
            # rather than the bar being made 97, so the photo keeps a whole
            # number of pixels per source row.
            canvas = Image.new("RGB", (1242, 376 + 100), PAPER)
            canvas.paste(photo.resize((1242, 376)), (0, 0))
            draw = ImageDraw.Draw(canvas)

            # Out of band: in the frame, but not what this section claims.
            # Drawing these in green was how a section captioned "0-10 m" ended
            # up showing objects at 20-30 m as though they were part of it.
            for item in truth:
                if item.label == "Pedestrian" and binner(item) != band:
                    box = item.box
                    draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                                   outline=GREY, width=1)

            for detection in visible:
                box = detection.box
                draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                               outline=BLUE, width=2)

            for item in found:
                if binner(item) != band:
                    continue
                box = item.box
                draw.rectangle([box.x1, box.y1, box.x2, box.y2],
                               outline=GREEN, width=2)

            for item, overlap in gone:
                if binner(item) != band:
                    continue
                box = item.box
                draw.rectangle([box.x1 - 3, box.y1 - 3, box.x2 + 3, box.y2 + 3],
                               outline=RED, width=3)
                # Say WHY it is a miss. "Boxed at 0.41 where 0.50 was needed"
                # and "nothing there at all" are different failures with
                # different fixes, and they looked identical before.
                note = f"{overlap:.2f} IoU" if overlap > 0.05 else "not seen"
                # Below the box when there is no room above it, and pulled left
                # of the right edge when the label would run off frame.
                width = draw.textlength(note, font=tiny)
                nx = min(max(2.0, box.x1 - 2), 1242 - width - 4)
                ny = box.y1 - 18 if box.y1 > 20 else min(box.y2 + 3, 376 - 17)
                draw.text((nx, ny), note, font=tiny, fill=RED)

            score = ap.get(band)
            bar = 376
            draw.text((22, bar + 6), f"Pedestrians at {band}", font=big, fill=INK)
            label = f"frame {frame}"
            draw.text((1242 - 22 - draw.textlength(label, font=small), bar + 14),
                      label, font=small, fill=(110, 116, 126))
            in_band = sum(1 for g in truth
                          if g.label == "Pedestrian" and binner(g) == band)
            draw.text((22, bar + 46),
                      (f"AP {score:.3f}    {in_band} in band"
                       if score is not None else ""),
                      font=small, fill=RED if (score or 0) < 0.1 else INK)
            draw.text((330, bar + 46),
                      "green found    red missed    blue detection    "
                      "grey another band",
                      font=small, fill=(150, 156, 166))
            draw.text((22, bar + 74),
                      f"boxes at confidence ≥ {args.score:g}; AP is over the "
                      f"whole precision-recall curve; a red box shows the best "
                      f"overlap achieved, {IOU:g} is needed",
                      font=tiny, fill=(110, 116, 126))


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
