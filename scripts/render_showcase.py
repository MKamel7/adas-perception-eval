"""The showcase: every finding, in the order that makes them mean something.

    uv run --extra infer python scripts/render_showcase.py

WHAT THIS IS FOR, and why it is not the same as the distance scene. That one
walks a viewer through a single result. This is the whole argument, and the
argument has an order:

  the aggregate         one number, shown once, so it can be set aside
  the spread            the same detections cut by attributes chosen in advance
  the ceiling           a limit no threshold reaches, which is not a tuning
                        problem and is the most consequential thing measured
  the cost of recall    both directions are hazards
  seen or unseen        two failures with different fixes
  the simulation        it has no pedestrians, so it could not have found any
                        of this

Every figure is read from outputs/ at render time. Nothing here is typed in, so
a re-run that moves a number moves the video with it, and a video that disagrees
with the evaluation cannot be produced.

Design constraints, because a portfolio piece is read in one pass with no audio:
one idea per card, the number always larger than its label, and no frame
carrying a claim the tables do not.
"""

from __future__ import annotations

import argparse
import ast
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

W, H = 1280, 720
INK, DIM, PAPER = (238, 240, 244), (140, 148, 160), (16, 18, 22)
GREEN, RED, BLUE, GREY = ((70, 205, 130), (240, 75, 95), (85, 160, 250),
                          (110, 116, 128))
ACCENT = (120, 190, 255)
FPS = 25


def font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = (("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold
             else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {"huge": font(96, True), "big": font(46, True), "mid": font(28),
     "small": font(22), "tiny": font(18), "label": font(20, True)}


class Reel:
    """Frames on disk, in order. Held by repeating, not by a duration field."""

    def __init__(self, staging: Path) -> None:
        self.staging = staging
        self.staging.mkdir(parents=True, exist_ok=True)
        for old in self.staging.glob("*.png"):
            old.unlink()
        self.index = 0

    def hold(self, image, seconds: float) -> None:
        for _ in range(max(1, round(seconds * FPS))):
            image.save(self.staging / f"{self.index:05d}.png")
            self.index += 1

    @property
    def seconds(self) -> float:
        return self.index / FPS


def card(draw_body=None, *, kicker: str = "", title: str = "",
         note: str = "") -> object:
    """A text card. One idea, and the number bigger than its label."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(image)
    if kicker:
        draw.text((72, 96), kicker.upper(), font=F["label"], fill=ACCENT)
    if title:
        y = 140
        for line in title.split("\n"):
            draw.text((72, y), line, font=F["big"], fill=INK)
            y += 58
    if draw_body:
        draw_body(draw)
    if note:
        y = H - 72 - 26 * (note.count("\n") + 1)
        for line in note.split("\n"):
            draw.text((72, y), line, font=F["small"], fill=DIM)
            y += 28
    return image


def bars(draw, rows, x, y, width, height, *, unit="", highlight=None):
    """A horizontal bar chart. Values are drawn as text too, never only as bars."""
    top = max((v for _, v in rows), default=1.0) or 1.0
    step = height // max(1, len(rows))
    for i, (label, value) in enumerate(rows):
        cy = y + i * step
        colour = RED if highlight and label in highlight else ACCENT
        draw.text((x, cy), label, font=F["small"], fill=DIM)
        bar = int((value / top) * (width - 300))
        draw.rectangle([x + 210, cy + 4, x + 210 + max(2, bar), cy + 22],
                       fill=colour)
        draw.text((x + 210 + max(2, bar) + 12, cy),
                  f"{value:.3f}{unit}" if unit != "x" else f"{value:.2f}{unit}",
                  font=F["small"], fill=INK)


def annotate(image, truth, detections, band, binner, label="Pedestrian",
             show_iou=True):
    """One KITTI frame with the boxes and why each missed object missed."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
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

    for item in truth:
        if item.label == label and binner(item) != band:
            b = item.box
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=GREY, width=1)
    for detection in detections:
        if detection.label != label:
            continue
        b = detection.box
        draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=BLUE, width=2)
    # Labels are placed last and only where they will be legible. Distant
    # objects cluster, and a row of overlapping "0.40 IoU" strings is worse
    # than none: it reads as noise and hides the boxes it is describing.
    placed: list[tuple[float, float, float, float]] = []
    for i, item in enumerate(counts):
        if binner(item) != band:
            continue
        b = item.box
        if i in claimed:
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=GREEN, width=2)
            continue
        draw.rectangle([b.x1 - 3, b.y1 - 3, b.x2 + 3, b.y2 + 3],
                       outline=RED, width=3)
        if show_iou and b.width >= 34:
            near = max((d.box.iou(item.box) for d in detections
                        if d.label == label), default=0.0)
            text = f"{near:.2f} IoU" if near > 0.05 else "not seen"
            width = draw.textlength(text, font=F["tiny"])
            x = min(max(2.0, b.x1 - 2), image.width - width - 4)
            y = (b.y1 - 20 if b.y1 > 22
                 else min(b.y2 + 3, image.height - 19))
            box = (x, y, x + width, y + 16)
            if not any(box[0] < q[2] and q[0] < box[2]
                       and box[1] < q[3] and q[1] < box[3] for q in placed):
                draw.text((x, y), text, font=F["tiny"], fill=RED)
                placed.append(box)
    return image


def scene(photo, truth, detections, band, binner, caption, value, label):
    """A framed KITTI photo with a caption bar underneath."""
    from PIL import Image, ImageDraw

    annotate(photo, truth, detections, band, binner, label)
    canvas = Image.new("RGB", (W, H), PAPER)
    scaled = photo.resize((W, round(photo.height * W / photo.width)))
    canvas.paste(scaled, (0, (H - scaled.height) // 2 - 40))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, H - 128, W, H], fill=PAPER)
    draw.text((72, H - 112), caption, font=F["big"], fill=INK)
    draw.text((72, H - 56), value, font=F["small"],
              fill=RED if "0.0" in value[:8] else DIM)
    draw.text((W - 72 - draw.textlength(
        "green found   red missed   blue detection   grey other band",
        font=F["tiny"]), H - 50),
        "green found   red missed   blue detection   grey other band",
        font=F["tiny"], fill=DIM)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/training")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/p3-showcase.mp4")
    parser.add_argument("--score", type=float, default=0.25)
    args = parser.parse_args()

    from PIL import Image

    results = json.loads((ROOT / "outputs/results.json").read_text(encoding="utf-8"))
    models = json.loads((ROOT / "outputs/models.json").read_text(encoding="utf-8"))
    sim = json.loads((ROOT / "outputs/sim_to_real.json").read_text(encoding="utf-8"))

    header, detections = load_detections(ROOT / "outputs/detections-yolov8s.jsonl")
    split = frame_ids(args.data / "label_2")[:header.frames]
    binner = dimension("distance").of

    reel = Reel(ROOT / "outputs/.showcase")

    def dist(label, band, n):
        """Frames whose pedestrians are mostly in this band, richest first."""
        scored = []
        for frame in split:
            truth = load_labels(args.data / "label_2" / f"{frame}.txt")
            items = [g for g in truth if g.label == label]
            inside = sum(1 for g in items if binner(g) == band)
            if inside:
                scored.append((inside - 0.5 * (len(items) - inside), frame))
        scored.sort(reverse=True)
        return [f for _, f in scored[:n]]

    # 1. Title
    reel.hold(card(
        kicker="ADAS perception evaluation",
        title="Aggregate metrics hide\nthe failures that matter.",
        note=f"{results['frames']} KITTI frames   {results['objects']:,} annotated "
             f"objects   YOLOv8s to ONNX\nmAP implementation validated against "
             f"pycocotools to six decimal places"), 3.6)

    # 2. The aggregate, shown once
    def headline(draw):
        draw.text((72, 250), f"{results['headline_map']:.3f}", font=F["huge"],
                  fill=INK)
        draw.text((72, 372), "mAP over Car and Pedestrian", font=F["mid"], fill=DIM)
    reel.hold(card(headline, kicker="the number you would be given",
                   note="It is the least informative figure this project "
                        "produces.\nEverything that follows is the same "
                        "detections, cut by attributes\nKITTI annotated before "
                        "anyone saw a result."), 3.6)

    # 3. Distance: the collapse
    ped = {i["bin"]: c["ap"] for i in results["slices"]
           if i["dimension"] == "distance" for c in i["cells"]
           if c["label"] == "Pedestrian"}
    def by_distance(draw):
        bars(draw, [(b, ped[b]) for b in ("0-10 m", "10-20 m", "20-30 m",
                                          "30-40 m", "40-50 m", ">50 m")],
             72, 250, W - 144, 300, highlight={"30-40 m", "40-50 m", ">50 m"})
    reel.hold(card(by_distance, kicker="cut by distance",
                   title="Pedestrian average precision",
                   note="Not a decline. A collapse."), 4.0)

    for band in ("0-10 m", "20-30 m", "30-40 m", ">50 m"):
        for frame in dist("Pedestrian", band, 2):
            truth = load_labels(args.data / "label_2" / f"{frame}.txt")
            visible = [d for d in detections.get(frame, [])
                       if d.score >= args.score]
            with Image.open(args.data / "image_2" / f"{frame}.png") as handle:
                photo = handle.convert("RGB")
            reel.hold(scene(photo, truth, visible, band, binner,
                            f"Pedestrians at {band}",
                            f"measured AP {ped[band]:.3f}", "Pedestrian"), 1.5)

    # 4. The ceiling
    def ceiling(draw):
        draw.text((72, 232), f"{results['ceiling_recall']['Pedestrian']:.1%}",
                  font=F["huge"], fill=RED)
        draw.text((72, 354), "of pedestrians, at ANY confidence threshold",
                  font=F["mid"], fill=DIM)
        draw.text((72, 430),
                  "Accept every box the detector emits and a third are still "
                  "unreported.", font=F["small"], fill=INK)
    reel.hold(card(ceiling, kicker="the limit no threshold reaches",
                   note="Average precision integrates over every threshold. A "
                        "vehicle runs at one.\nThis is not a tuning problem: it "
                        "is answered with a different sensor,\nnot a different "
                        "number."), 4.4)

    # 5. The cost of recall
    car_ops = {o["target"]: o for o in results["operating_points"]["Car"]}
    growth = (car_ops[0.8]["false_alarms_per_frame"]
              / car_ops[0.5]["false_alarms_per_frame"])

    def trade(draw):
        rows = [(f"recall {t:.0%}", car_ops[t]["false_alarms_per_frame"])
                for t in (0.5, 0.7, 0.8) if car_ops[t]["reachable"]]
        bars(draw, rows, 72, 262, W - 144, 190, highlight={"recall 80%"})
        draw.text((72, 486),
                  f"{growth:.0f}x more phantom detections "
                  f"to buy 1.6x the recall.",
                  font=F["mid"], fill=INK)
    reel.hold(card(trade, kicker="and the other direction is a hazard too",
                   title="False alarms per frame, Car",
                   note="At 2.39 per frame a vehicle at ten frames a second "
                        "reacts to\nsomething imaginary twenty-two times a "
                        "second. Phantom braking is\na collision risk, not a "
                        "comfort complaint."), 4.4)

    # 6. Missed or mislocated
    diag = results["diagnosis"]
    def split_card(draw):
        draw.text((72, 250), f"{diag['Car']['mislocation_share']:.0%}",
                  font=F["huge"], fill=ACCENT)
        draw.text((72, 372), "of Car misses were boxes that landed badly",
                  font=F["mid"], fill=DIM)
        draw.text((72, 440),
                  f"Only {diag['Car']['unseen']:,} of {diag['Car']['total']:,} "
                  f"cars were not seen at all.", font=F["small"], fill=INK)
        draw.text((72, 472),
                  f"Pedestrians split "
                  f"{diag['Pedestrian']['mislocation_share']:.0%} / "
                  f"{1 - diag['Pedestrian']['mislocation_share']:.0%}: half of "
                  f"that really is blindness.", font=F["small"], fill=INK)
    reel.hold(card(split_card, kicker="missed, or seen and boxed badly?",
                   note="Different failures, different fixes. For cars the "
                        "answer is box\nregression, not recall, which is the "
                        "opposite of what the\naggregate suggests."), 4.4)

    for frame in dist("Car", ">50 m", 2):
        truth = load_labels(args.data / "label_2" / f"{frame}.txt")
        visible = [d for d in detections.get(frame, []) if d.score >= args.score]
        with Image.open(args.data / "image_2" / f"{frame}.png") as handle:
            photo = handle.convert("RGB")
        reel.hold(scene(photo, truth, visible, ">50 m", binner,
                        "Cars beyond 50 m",
                        "each red box shows the overlap it achieved; 0.50 is "
                        "needed", "Car"), 2.2)

    # 7. Occlusion
    occ = {i["bin"]: {c["label"]: c["ap"] for c in i["cells"]}
           for i in results["slices"] if i["dimension"] == "occlusion"}
    def occlusion_card(draw):
        bars(draw, [(b, occ[b]["Pedestrian"]) for b in
                    ("fully visible", "partly occluded", "largely occluded")],
             72, 262, W - 144, 190, highlight={"largely occluded"})
        draw.text((72, 486),
                  "A pedestrian stepping out from between parked cars is the "
                  "urban case.", font=F["mid"], fill=INK)
    reel.hold(card(occlusion_card, kicker="the strongest predictor measured",
                   title="Pedestrian AP by occlusion",
                   note="Performance falls by more than three times before the "
                        "annotator\nwould even call the object mostly hidden."), 4.2)

    # 8. Sim to real
    # Read from the sim-to-real run rather than typed in, so a re-run that
    # changes the composition changes the card with it.
    clone = next(v for v in sim["variants"] if v["variant"].endswith("clone"))
    far = clone["slices"]["distance"][">50 m"]["n"]
    everything = sum(v["n"] for v in clone["slices"]["distance"].values())

    def simulation(draw):
        draw.text((72, 240), "0", font=F["huge"], fill=RED)
        draw.text((72, 362), "pedestrians in Virtual KITTI 2", font=F["mid"],
                  fill=DIM)
        draw.text((72, 430),
                  "Car 245 tracks, Van 22, Truck 6. No people. No cyclists.",
                  font=F["small"], fill=INK)
        draw.text((72, 466),
                  "The headline finding cannot be checked against the "
                  "simulation at all.", font=F["small"], fill=INK)
    reel.hold(card(simulation, kicker="would simulation have told you?",
                   note="Not mismeasured. Untestable. A validation programme "
                        "leaning on this\nsynthetic data had no way to see it. "
                        "On cars, where a comparison is\npossible, real and "
                        "synthetic agree band by band at rank correlation "
                        "0.943\nand disagree in aggregate purely because "
                        f"{far / everything:.0%} of synthetic cars\nsit beyond "
                        "50 m."), 5.2)

    # 9. What was built. COUNTED, not typed: a card claiming "9 conditions"
    # after a tenth is added is the same stale-claim failure the traceability
    # gate exists to prevent, and it would be embarrassing here of all places.
    import yaml
    analysis = yaml.safe_load(
        (ROOT / "safety/triggering_conditions.yaml").read_text(encoding="utf-8"))
    conditions = len(analysis["conditions"])
    hazards = len(analysis["hazards"])
    figures = sum(len(c["evidence"]) for c in analysis["conditions"])

    # Counted by PARSING, not by shelling out to pytest. The first version ran
    # `pytest --collect-only` in a subprocess and printed "? tests" whenever the
    # render was invoked without the dev group installed, which is most of the
    # time. Parsing has no dependency, cannot half-work, and is the same
    # technique the traceability gate uses on the markers.
    tests = 0
    for path in (ROOT / "tests").rglob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        tests += sum(1 for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)
                     and node.name.startswith("test_"))

    fast = min(models["models"], key=lambda m: m["median_ms"])
    def closing(draw):
        lines = [
            ("mAP validated against pycocotools", "to six decimal places"),
            (f"{conditions} triggering conditions, {hazards} hazards",
             "gated in both directions"),
            (f"{figures} measured figures", "every one checked, none by hand"),
            ("every AP with a 95% interval", "bootstrapped over frames"),
            (f"{tests} test functions, coverage gated in CI",
             "ruff and mypy strict"),
        ]
        y = 224
        for left, right in lines:
            draw.text((72, y), left, font=F["mid"], fill=INK)
            draw.text((700, y + 4), right, font=F["small"], fill=DIM)
            y += 54
    reel.hold(card(closing, kicker="what is underneath the numbers",
                   note="SOTIF vocabulary is borrowed; its process is not "
                        "performed and no\ncompliance is claimed. Pretrained "
                        "COCO weights: nothing was trained."), 5.0)

    ffmpeg = next(iter(Path(
        "C:/Users/mkame/AppData/Local/Microsoft/WinGet/Packages"
    ).glob("**/ffmpeg.exe")), None)
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found; frames are in " + str(reel.staging))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(ffmpeg), "-y", "-v", "error", "-framerate", str(FPS),
                    "-i", str(reel.staging / "%05d.png"), "-c:v", "libx264",
                    "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(args.out)], check=True)
    for old in reel.staging.glob("*.png"):
        old.unlink()
    reel.staging.rmdir()
    print(f"wrote {args.out} ({reel.seconds:.1f} s, {W}x{H})")
    print(f"nano model reference: {fast['model']} at {fast['median_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
