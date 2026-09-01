"""Evaluate the cached detections and write the report.

    uv run --extra report python scripts/evaluate.py

One command, from a fresh checkout plus data, to a self-contained HTML file that
can be committed and opened without a server. No notebook, no dashboard, no
build step: the artifact IS the deliverable and a reader should not have to run
anything to see it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ape.cache import load_detections  # noqa: E402
from ape.classes import EVALUATED, HEADLINE  # noqa: E402
from ape.evaluate import IOU, evaluate, operating_table  # noqa: E402
from ape.kitti import frame_ids, load_labels  # noqa: E402
from ape.metrics import false_negative_rate, recall_at_precision  # noqa: E402
from ape.outcomes import FAILURES  # noqa: E402
from ape.report import render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path,
                        default=ROOT / "outputs/detections-yolov8s.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/training/label_2")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/report.html")
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/results.json")
    args = parser.parse_args()

    header, detections = load_detections(args.detections)
    print(f"{header.model}, {header.frames} frames, "
          f"score >= {header.score_threshold}")

    # The SPLIT, reconstructed the same way run_inference.py chose it, and not
    # the set of frames that happen to appear in the cache.
    #
    # Taking the frames from the detections is the obvious shortcut and it is
    # wrong: a frame where the detector found nothing at all has no entry in the
    # cache, so it silently leaves the evaluation, and every object in it leaves
    # the recall denominator with it. A detector that returns nothing on the
    # hardest frames would then be rewarded for it. This cost two frames and
    # 0.0008 of Car AP before it was caught.
    split = frame_ids(args.labels)[:header.frames]
    if len(split) != header.frames:
        raise SystemExit(
            f"the cache says {header.frames} frames but only {len(split)} are "
            f"available under {args.labels}")
    truth = {frame: load_labels(args.labels / f"{frame}.txt") for frame in split}

    missing = sorted(set(detections) - set(truth))
    if missing:
        raise SystemExit(
            f"{len(missing)} frames have detections but are outside the split "
            f"({missing[:3]}...). The cache and the labels disagree about what "
            f"was evaluated.")
    silent = sorted(set(truth) - set(detections))
    if silent:
        print(f"  {len(silent)} frames produced no detections at all "
              f"({', '.join(silent[:4])}); they are evaluated as misses, "
              f"not dropped")

    result = evaluate(detections, truth, IOU)

    print(f"\nframes {result.frames}, objects {result.objects}, IoU {IOU}")
    print(f"{'class':<12} {'AP':>8} {'95% interval':>18} {'positives':>10} "
          f"{'recall':>8}")
    for label in EVALUATED:
        curve = result.overall[label]
        ci = result.overall_interval.get(label)
        band = f"[{ci.low:.3f}, {ci.high:.3f}]" if ci else ""
        marker = "" if label in HEADLINE else "  (mapping-limited)"
        print(f"{label:<12} {curve.average_precision:>8.4f} {band:>18} "
              f"{curve.positives:>10} {curve.best_recall:>8.3f}{marker}")
    print(f"\nheadline mAP over {', '.join(HEADLINE)}: {result.headline:.4f}")

    for label in HEADLINE:
        best, worst, where_best, where_worst = result.spread(label)
        if best == best:
            # A ratio against a slice that scored zero is not a number, it is a
            # division artefact. "10229x" was printed once and means nothing
            # beyond "the worst slice found nothing at all", which is the more
            # useful sentence anyway.
            spread = (f"spread {best / worst:.1f}x" if worst > 0.005
                      else "the worst slice found essentially nothing")
            print(f"  {label}: best {best:.3f} ({where_best}), "
                  f"worst {worst:.3f} ({where_worst}), {spread}")

    print("\nwas it missed, or just boxed badly?")
    for label in HEADLINE:
        diagnosis = result.diagnosis[label]
        aps = "  ".join(f"AP@{k:g} {v[label]:.3f}"
                        for k, v in sorted(result.at_iou.items()))
        print(f"  {label:<11} {aps}")
        print(f"              {diagnosis.found_tight} found, "
              f"{diagnosis.mislocated} mislocated, {diagnosis.unseen} unseen "
              f"-> {diagnosis.mislocation_share:.0%} of misses are a box problem")

    print("\nwhat kind of mistake was it? (AP counts every wrong box the same)")
    for label in HEADLINE:
        breakdown = result.outcomes[label]
        parts = "  ".join(f"{outcome.value} {breakdown.counts[outcome]}"
                          f" ({breakdown.share(outcome):.0%})"
                          for outcome in FAILURES)
        print(f"  {label:<11} {breakdown.false_positives} false positives")
        print(f"              {parts}")

    print("\nbeyond AP: what no threshold choice can buy")
    for label in HEADLINE:
        curve = result.overall[label]
        print(f"  {label:<11} false-negative rate "
              f"{false_negative_rate(curve):.1%} at the recall ceiling")
        print(f"              recall at 90% precision "
              f"{recall_at_precision(curve, 0.90):.1%}, "
              f"at 50% precision {recall_at_precision(curve, 0.50):.1%}")

    print("\nchoosing an operating point (AP integrates over all of them; "
          "a vehicle runs at one)")
    for label in HEADLINE:
        print(f"  {label}: ceiling {result.ceiling.get(label, 0):.1%} recall "
              f"at any threshold")
        for target, point in operating_table(result, label):
            if point is None:
                print(f"    {target:>5.0%}  UNREACHABLE")
            else:
                print(f"    {target:>5.0%}  threshold {point.threshold:.3f}  "
                      f"precision {point.precision:.3f}  "
                      f"{point.false_alarms_per_frame:.2f} false alarms/frame")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({
        "model": header.model, "frames": result.frames,
        "objects": result.objects, "iou": IOU,
        "headline_map": result.headline,
        "overall": {k: v.average_precision for k, v in result.overall.items()},
        "overall_ci": {k: [v.low, v.high]
                       for k, v in result.overall_interval.items()},
        "by_difficulty": {tier: {k: v.average_precision for k, v in cls.items()}
                          for tier, cls in result.by_difficulty.items()},
        "ceiling_recall": result.ceiling,
        "at_iou": {str(k): v for k, v in result.at_iou.items()},
        "diagnosis": {k: {"total": d.total, "found": d.found_tight,
                          "mislocated": d.mislocated, "unseen": d.unseen,
                          "mislocation_share": d.mislocation_share}
                      for k, d in result.diagnosis.items()},
        "false_positive_kinds": {
            k: {"total": b.false_positives,
                "counts": {o.value: b.counts[o] for o in FAILURES},
                "shares": {o.value: b.share(o) for o in FAILURES}}
            for k, b in result.outcomes.items()},
        "beyond_ap": {
            k: {"false_negative_rate": false_negative_rate(c),
                "max_recall": c.best_recall,
                "recall_at_precision_90": recall_at_precision(c, 0.90),
                "recall_at_precision_50": recall_at_precision(c, 0.50)}
            for k, c in result.overall.items()},
        "operating_points": {
            label: [{"target": target,
                     "threshold": p.threshold if p else None,
                     "precision": p.precision if p else None,
                     "false_alarms_per_frame": (p.false_alarms_per_frame
                                                if p else None),
                     "missed": p.missed if p else None,
                     "reachable": p is not None}
                    for target, p in operating_table(result, label)]
            for label in EVALUATED},
        "slices": [{"dimension": s.dimension, "bin": s.bin,
                    "cells": [{"label": c.label,
                               "ap": c.curve.average_precision,
                               "positives": c.curve.positives,
                               "trustworthy": c.trustworthy,
                               "ci_low": c.interval.low if c.interval else None,
                               "ci_high": c.interval.high if c.interval else None,
                               "frames": c.interval.frames if c.interval else 0}
                              for c in s.cells]}
                   for s in result.slices],
    }, indent=2), encoding="utf-8")

    args.out.write_text(render(result, header), encoding="utf-8")
    print(f"\nwrote {args.out}\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
