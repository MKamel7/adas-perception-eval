# ADAS perception evaluation

**Aggregate metrics hide the failures that matter.** A detector reported at 0.68
mAP can be near-blind to occluded pedestrians beyond 40 metres and the single
number will never say so. This repository is the harness that says so: slice-based
detection metrics on KITTI, with the mAP implementation validated against the
reference COCO implementation rather than trusted, and the worst slices written up
as triggering conditions in ISO 21448 (SOTIF) vocabulary.

**It is an evaluation project, not a model project. Nothing here is trained.** The
detector is a pretrained commodity. What is built is the thing that decides
whether a detector is good enough, which is what perception validation teams
actually spend their days on.

## Status

**Milestones 1, 3, 4 and 7 are done. Milestone 2 is done and MISSED its
performance criterion.** Milestones 5, 6 and 8 are not started. The criteria were
written before any code, so a result cannot be rationalised into a pass
afterwards, and that cuts both ways: M2's budget is missed and is recorded as
missed rather than adjusted.

## The result

1500 KITTI frames, 7945 annotated objects, YOLOv8s exported to ONNX, IoU 0.5.

| class | AP | objects | note |
|---|---|---|---|
| Car | **0.754** | 5579 | |
| Pedestrian | **0.495** | 893 | |
| Cyclist | 0.006 | 341 | mapping-limited, excluded from the headline |

**Headline mAP over Car and Pedestrian: 0.625.**

That number is the least useful thing on this page, which is the entire point.
Cut the same detections along attributes KITTI annotated before anyone saw a
result:

| | Car | Pedestrian |
|---|---|---|
| easy | 0.938 | 0.691 |
| moderate | 0.882 | 0.594 |
| hard | 0.772 | 0.501 |
| 0-10 m | 0.900 | 0.689 |
| 30-40 m | 0.518 | **0.007** |
| 40-50 m | 0.341 | **0.001** |
| >50 m | 0.172 | **0.000** |
| largely occluded | 0.250 | 0.023 |

**A pedestrian detector reported at 0.495 is effectively blind beyond 30 metres.**
Not degraded, blind: AP 0.007 over 74 objects, and zero beyond 50 m. At 50 km/h a
car covers 30 m in about two seconds. That is the finding, and no aggregate
number contains it.

Car spreads 5.5x between its best slice and its worst. The spec predicted 2x to
3x before running, so the prediction understated the effect; that is recorded
here rather than quietly updated.

## Why calibration is in a 2D detection benchmark

Because KITTI annotates each object twice, in two different spaces: a 3D position
in camera coordinates and a 2D box in the image. Those two are redundant, and
redundancy is testable.

Project the 3D centre through the calibration and it must land inside the 2D box.
A transposed matrix, a dropped rectification or a flipped sign breaks that
immediately, against real data, with no hand-built fixture to maintain. Frame
conventions are exactly where this gets quietly wrong, and the failure is silent:
every object simply sits a few metres from where it really was, every distance
slice inherits the error, and nothing ever raises.

The test that checks it is paired with a second test that deliberately flips a
sign and confirms the first one fails. A consistency check nobody has watched fail
is an assumption, not evidence.

### And that guard has a measured limit, not an assumed one

The calibration code reached 100% branch coverage, which said nothing useful. So
the calibration was broken six different ways to find out which breakages the
check would actually notice:

| Injected fault | Caught |
|---|---|
| A sign flipped in the projection | yes, 53 of 58 objects leave their box |
| u and v swapped | yes, all 58 |
| The baseline translation negated | yes |
| **Rectification skipped entirely** | **no** |
| **R0_rect transposed** | **no** |

The misses are not a bug. KITTI's R0_rect is under one degree from identity, so
dropping it shifts a projected point by a fraction of a box, which is well inside
the slack that "the centre is somewhere in the box" allows. No threshold fixes
this: with rectification dropped the worst offset is *smaller* than with the
calibration intact. It is a property of the data.

Two things follow. Rectification is proven to be applied by a separate test using
a synthetic quarter turn, since the real data cannot show it. And the two blind
spots are asserted **in the failing direction**, so if a future dataset makes them
observable the suite fails and the limit gets revisited rather than inherited as
folklore.

The first version of this README claimed the check caught a dropped
rectification. The sweep is what said otherwise. This is the same lesson as the
[virtual production cell](https://github.com/MKamel7/virtual-production-cell):
coverage measures which lines ran, not which situations were imagined.

## What is being measured, and against what

| Layer | Choice | What it beat, and why |
|---|---|---|
| Inference | ONNX Runtime, CPU | PyTorch. ONNX is what ships to embedded automotive targets, it is several times faster on CPU, and the export step is itself the industry-relevant part |
| Detector | pretrained YOLOv8, exported | training anything. The point is the harness, and a model trained on KITTI would only inflate the score |
| Metric reference | `pycocotools` | trusting my own mAP. The comparison **is** the deliverable |
| Report | Jinja2 to a self-contained HTML file | a dashboard or a notebook. No server, no build step, the artifact is committed and readable |

`pycocotools` is a dependency of the **tests**, never of the pipeline. Nothing in
the measurement path may import it, or the validation would be circular.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Ingest and calibration | **done.** Labels parse; projection verified by a test that catches a deliberate sign flip, on a committed 20-frame fixture |
| M2 | Inference | **done, criterion missed.** Export is reproducible and the class mapping is data. 1500 frames took **13 minutes against a 5 minute budget**, see below |
| M3 | Metrics validated | **done.** Own AP agrees with `pycocotools` **exactly, to six decimal places**, against a required 0.001 |
| M4 | Slicing | **done.** Six dimensions, every one from a ground-truth attribute, all committed before the run |
| M5 | Sim-to-real | not started. Virtual KITTI 2 is 7.5 GB and is not yet downloaded |
| M6 | Taxonomy | not started. The slice evidence it needs now exists |
| M7 | Report | **done.** One command produces `outputs/report.html` |
| M8 | Demo | not started |

### M2 missed its budget, and the reason is not the code

1500 frames took 799 seconds against a 300 second budget. I assumed the
bottleneck was my non-maximum suppression, rewrote it from a Python loop to
numpy, and the run went from 844 seconds to 799. Profiling afterwards, which
should have come first, put the time where it actually was:

| stage | per frame | share |
|---|---|---|
| ONNX forward | 397 ms | **83%** |
| preprocess | 35 ms | 7% |
| image load | 34 ms | 7% |
| postprocess (what I rewrote) | 12 ms | **2.4%** |

The budget is missed by YOLOv8s on a 15 W mobile CPU, not by the pipeline
around it. YOLOv8n would fit the budget at a cost in accuracy; that trade has
not been made because the accuracy is what is being measured. The criterion was
optimistic when it was written and it is left standing, marked as missed.

## Expected results, recorded before running

- **Cars around 0.5 to 0.7 mAP@0.5**, pedestrians and cyclists materially worse.
  Anything above 0.9 means the ground-truth handling is wrong, not that the
  detector is excellent.
- **The headline is the spread, not the aggregate.** A 2x to 3x gap between best
  and worst slice is expected, with far occluded pedestrians as the floor.
- **Metric agreement within 0.001, or there is a bug.** No tolerance argument.
- **Sim-to-real: the shapes probably rhyme and the absolutes will not.** If
  synthetic data degrades in the same order as real data, simulation-based
  validation has some predictive value here. If it does not, that is the more
  interesting finding and it gets reported just as loudly.

## Honest limits

- Pretrained COCO weights evaluated on KITTI classes. A KITTI-trained model would
  score higher. The score is not the deliverable.
- 2D only. Real ADAS validation is 3D and multi-sensor.
- KITTI is daytime, fair weather, one city, one sensor rig. The weather and
  lighting evidence comes from synthetic data, which is a weaker claim and is
  labelled as such wherever it appears.
- No ODD definition, no exposure or controllability analysis, no residual risk
  argument. **SOTIF's vocabulary is borrowed on purpose; its process is not
  performed, and no compliance is claimed.**
- CARLA was excluded on hardware grounds, not preference: it wants an NVIDIA GPU
  with at least 6 GB of VRAM against 1 GB of shared integrated graphics here.
- CarMaker, dSPACE and aiSim are what industry actually runs, and all are
  commercially licensed. Scenario simulation and recorded-data evaluation are two
  halves of one job; this project does the half that needs no licence.
- CI runs the metric implementations against small committed fixtures. It does not
  run the full evaluation, because KITTI is gigabytes and a CI job claiming
  otherwise would be lying.

## Getting the data

KITTI is not redistributed here. `data/` is gitignored; the committed 20-frame
fixture under `tests/fixtures/` is what the test suite runs against.

```
uv sync --group dev
uv run pytest
```

## Licence

Code under MIT. KITTI is CC BY-NC-SA 3.0 and is not included.
