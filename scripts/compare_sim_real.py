"""M5: does the simulation degrade the way reality does?

    uv run --extra infer python scripts/compare_sim_real.py

THE QUESTION, and it is not "do the scores match". They will not: synthetic
imagery is cleaner, the annotation is perfect, and an absolute comparison
between the two is meaningless. The question is whether the SHAPE of the
degradation rhymes. If the things that are hard in simulation are the things
that are hard in reality, simulation-based validation has predictive value here.
If they are not, that is the more interesting finding and it gets reported just
as loudly.

THE ANSWER IS ALREADY PARTLY KNOWN, before any inference runs, and it is the
strongest result in this milestone:

    Virtual KITTI 2 contains no pedestrians. Car, Van and Truck only.

This project's headline finding is that pedestrian detection collapses beyond
30 metres. A validation programme relying on this synthetic data could not have
found that, not because it would have got the number wrong but because it has no
pedestrians to be wrong about. That is reported first, because a correlation
computed on cars would otherwise imply a confidence the data does not support.

What is compared, therefore, is Car degradation by distance and by occlusion,
across the baseline render and the weather and lighting variants that KITTI
itself has no examples of.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.classes import neutral_labels  # noqa: E402
from ape.detect import Detector  # noqa: E402
from ape.evaluate import IOU  # noqa: E402
from ape.match import assign, in_slice  # noqa: E402
from ape.metrics import average_precision  # noqa: E402
from ape.slices import dimension  # noqa: E402
from ape.vkitti import Variant, image_path, load_labels, variants  # noqa: E402

#: Only the dimensions both datasets can express. Truncation is comparable,
#: occlusion is NOT strictly comparable because the bridge from Virtual KITTI's
#: continuous occupancy to KITTI's three levels is invented, so it is reported
#: with that caveat rather than dropped.
COMPARED = ("distance", "box height", "occlusion")


def evaluate_variant(detector: Detector, root: Path, variant: Variant,
                     data_root: Path, limit: int) -> dict:
    truth = load_labels(root, variant)
    frames = sorted(truth)[:limit]
    truth = {f: truth[f] for f in frames}

    detections: dict[str, list] = {}
    for frame in frames:
        image = image_path(data_root, variant, frame)
        if not image.exists():
            raise SystemExit(
                f"{image} is missing. Run scripts/fetch_vkitti.py; evaluating "
                f"the frames that happen to be present would report a number "
                f"for a split nobody chose.")
        detections[frame] = detector.detect(image, frame)

    neutral = neutral_labels("Car")
    out: dict = {"variant": variant.key, "frames": len(frames),
                 "objects": sum(len(v) for v in truth.values()), "slices": {}}
    out["overall"] = average_precision(
        assign(detections, truth, "Car", neutral, IOU)).average_precision

    for name in COMPARED:
        dim = dimension(name)
        out["slices"][name] = {}
        for bin_name in dim.bins:
            curve = average_precision(assign(detections, truth, "Car", neutral,
                                             IOU, in_slice(dim.of, str(bin_name))))
            if curve.positives >= 10:
                out["slices"][name][str(bin_name)] = {
                    "ap": curve.average_precision, "n": curve.positives}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/yolov8s.onnx")
    parser.add_argument("--vkitti", type=Path, default=ROOT / "data/vkitti")
    parser.add_argument("--real", type=Path, default=ROOT / "outputs/results.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/sim_to_real.json")
    parser.add_argument("--scene", default="Scene01")
    parser.add_argument("--limit", type=int, default=120,
                        help="frames per variant; six variants at this size is "
                             "already an hour of CPU inference")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()

    available = [v for v in variants(args.vkitti) if v.scene == args.scene]
    if not available:
        raise SystemExit(f"no {args.scene} under {args.vkitti}")

    detector = Detector(args.model, score_threshold=0.05, threads=args.threads)
    real = json.loads(args.real.read_text(encoding="utf-8"))

    print(f"THE FINDING THAT NEEDS NO INFERENCE:\n"
          f"  Virtual KITTI 2 annotates Car, Van and Truck. It contains no\n"
          f"  pedestrians, so this project's headline result cannot be checked\n"
          f"  against it at all.\n")

    results = []
    for variant in available:
        print(f"{variant.key} ...", flush=True)
        outcome = evaluate_variant(detector, args.vkitti, variant,
                                   args.vkitti, args.limit)
        results.append(outcome)
        print(f"  Car AP {outcome['overall']:.3f} over {outcome['objects']} objects")

    baseline = next((r for r in results if r["variant"].endswith("clone")), None)

    print(f"\n{'variant':<22} {'Car AP':>8} {'vs clone':>10}")
    for outcome in results:
        delta = (f"{outcome['overall'] - baseline['overall']:+.3f}"
                 if baseline else "-")
        print(f"{outcome['variant']:<22} {outcome['overall']:>8.3f} {delta:>10}")

    # The shape question: does synthetic Car degrade with distance in the same
    # ORDER as real Car, even though the absolute numbers differ?
    if baseline:
        real_distance = {}
        for item in real["slices"]:
            if item["dimension"] == "distance":
                for cell in item["cells"]:
                    if cell["label"] == "Car" and cell["positives"] >= 10:
                        real_distance[item["bin"]] = cell["ap"]
        synthetic = {k: v["ap"] for k, v in baseline["slices"]["distance"].items()}
        shared = [b for b in real_distance if b in synthetic]

        print(f"\nCar by distance, real against synthetic baseline:")
        print(f"{'band':<12} {'real':>8} {'synthetic':>10}")
        for band in shared:
            print(f"{band:<12} {real_distance[band]:>8.3f} {synthetic[band]:>10.3f}")

        real_order = sorted(shared, key=lambda b: real_distance[b])
        synth_order = sorted(shared, key=lambda b: synthetic[b])
        agrees = real_order == synth_order
        print(f"\nWorst-to-best ordering agrees: {agrees}")
        print(f"  real:      {' < '.join(real_order)}")
        print(f"  synthetic: {' < '.join(synth_order)}")

    args.out.write_text(json.dumps({
        "no_pedestrians_in_simulation": True,
        "variants": results,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
