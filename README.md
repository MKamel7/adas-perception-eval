# ADAS perception evaluation

[![CI](https://github.com/MKamel7/adas-perception-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/MKamel7/adas-perception-eval/actions)
[![Dataset](https://img.shields.io/badge/dataset-KITTI-orange)](https://www.cvlibs.net/datasets/kitti/)
[![Validated against](https://img.shields.io/badge/mAP-validated%20vs%20pycocotools-brightgreen)](tests)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue)](LICENSE)


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

![slice-based detection on KITTI](outputs/p3-showcase.gif)

*Detections as the harness sees them. The point of the project is not the boxes, it is
knowing which slices of the data they fail on.*

## 🚦 Status

**All eight milestones are done. Milestone 2 MISSED its performance criterion**
and is recorded as missed rather than adjusted to fit the result. The criteria were
written before any code, so a result cannot be rationalised into a pass
afterwards, and that cuts both ways: M2's budget is missed and is recorded as
missed rather than adjusted.

## 🛠️ Built with

| | |
| --- | --- |
| **Language** | Python |
| **Inference** | ONNX Runtime |
| **Metrics** | Slice-based mAP, validated against pycocotools |
| **Data** | KITTI 2D object detection |
| **Engineering** | Expected results recorded before running, GitHub Actions CI |

## 📊 The result

**The full KITTI training split: 7481 frames, 40,570 annotated objects.** YOLOv8s exported to ONNX, IoU 0.5.

| class | AP | objects | note |
|---|---|---|---|
| Car | **0.754** [0.750, 0.762] | 28742 | |
| Pedestrian | **0.506** [0.488, 0.523] | 4487 | |
| Cyclist | 0.008 [0.005, 0.013] | 1627 | mapping-limited, excluded from the headline |

**Headline mAP over Car and Pedestrian: 0.630.**

That number is the least useful thing on this page, which is the entire point.
Cut the same detections along attributes KITTI annotated before anyone saw a
result:

| | Car | Pedestrian |
|---|---|---|
| easy | 0.937 | 0.686 |
| moderate | 0.880 | 0.593 |
| hard | 0.765 | 0.517 |
| 0-10 m | 0.866 | 0.709 |
| 10-20 m | 0.793 | 0.352 |
| 20-30 m | 0.668 | 0.077 |
| 30-40 m | 0.501 | **0.009** |
| 40-50 m | 0.326 | **0.001** |
| >50 m | 0.164 | **0.000** |
| largely occluded | 0.267 | 0.026 |

Every figure carries a **95% confidence interval from bootstrapping frames**,
not objects: people standing in one group are not independent observations, and
resampling objects would understate the range. Car overall is 0.754 [0.750,
0.762]; Pedestrian is 0.506 [0.488, 0.523]. The headline claim is tested against
its own uncertainty rather than asserted: the 0-10 m and 30-40 m pedestrian
intervals do not overlap, so the difference is more than the sample explains.
Overlapping intervals are *not* evidence of no difference, and the report says
so rather than letting an overlap read as a null result.

Distance here is **depth along the optical axis**, not radial distance to the
object. That is the quantity time-to-collision depends on, and the choice
matters: 9.7% of pedestrians would fall in a different band under the other
definition, so it is stated rather than left implicit.

**A pedestrian detector reported at 0.506 is effectively blind beyond 30 metres.**
Not degraded, blind: AP 0.009 at 30-40 m, and 0.000 beyond 50 m. At 50 km/h a
car covers 30 m in about two seconds. That is the finding, and no aggregate
number contains it.

Car spreads 5.7x between its best slice and its worst. The spec predicted 2x to
3x before running, so the prediction understated the effect; that is recorded
here rather than quietly updated.

## 🖼️ The showcase

`outputs/p3-showcase.mp4` is 55 seconds covering the whole argument in the order
that makes it mean something: the aggregate shown once so it can be set aside,
the spread, the recall ceiling no threshold reaches, the price of buying recall,
the split between missed and mislocated, and the simulation that has no
pedestrians to be blind to.

**Every figure in it is read from `outputs/` at render time.** Nothing is typed
in, including the counts on the closing card, so a re-run that moves a number
moves the video with it and a video that disagrees with the evaluation cannot be
produced. `outputs/p3-demo.mp4` is the shorter 21 s piece on the distance
finding alone.

## 🎚️ Why calibration is in a 2D detection benchmark

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

## 📏 What is being measured, and against what

| Layer | Choice | What it beat, and why |
|---|---|---|
| Inference | ONNX Runtime, CPU | PyTorch. ONNX is what ships to embedded automotive targets, it is several times faster on CPU, and the export step is itself the industry-relevant part |
| Detector | pretrained YOLOv8, exported | training anything. The point is the harness, and a model trained on KITTI would only inflate the score |
| Metric reference | `pycocotools` | trusting my own mAP. The comparison **is** the deliverable |
| Report | a self-contained HTML file | a dashboard or a notebook. No server, no build step, the artifact is committed and readable |
| Uncertainty | frame-level bootstrap, 400 resamples | an object count. Ten objects in ten frames and ten in one frame are not equally informative and a count cannot tell them apart |
| CI | ruff, mypy strict, full suite at 90% coverage, traceability gate | nothing. CI cannot run the evaluation, since the data is 20 GB, and a workflow claiming otherwise would be lying |

`pycocotools` is a dependency of the **tests**, never of the pipeline. Nothing in
the measurement path may import it, or the validation would be circular.

## 🗳️ Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Ingest and calibration | **done.** Labels parse; projection verified by a test that catches a deliberate sign flip, on a committed 20-frame fixture |
| M2 | Inference | **done, criterion missed.** Export is reproducible and the class mapping is data. 1500 frames took **13 minutes against a 5 minute budget**, see below |
| M3 | Metrics validated | **done.** Own AP agrees with `pycocotools` **exactly, to six decimal places**, against a required 0.001 |
| M4 | Slicing | **done.** Six dimensions, every one from a ground-truth attribute, all committed before the run |
| M5 | Sim-to-real | **done.** Six renders of Scene01, rank correlation 0.943 against real, and the finding below |
| M6 | Taxonomy | **done.** Six triggering conditions, three hazards, seven demonstrating tests, gated in both directions |
| M7 | Report | **done.** One command produces `outputs/report.html` |
| M8 | Demo and README | **done.** A 55 s showcase covering every finding, a 21 s distance scene, example frames chosen by rule, README |

### Which model, and what does the smaller one cost?

Both models were run over all 7481 frames, same order, same preprocessing,
scored by the same code. Latency measured separately on a warm session, since a
first call includes graph optimisation no steady-state deployment pays.

| model | size | median | p95 | 1500 frames | budget |
|---|---|---|---|---|---|
| YOLOv8n | 12.8 MB | 378 ms | 424 ms | 567 s | **missed** |
| YOLOv8s | 44.8 MB | 581 ms | 647 ms | 872 s | **missed** |

| model | Car | Pedestrian | headline | Pedestrian recall ceiling |
|---|---|---|---|---|
| YOLOv8n | 0.720 | 0.458 | 0.589 | 64.9% |
| YOLOv8s | 0.754 | 0.506 | 0.630 | 68.7% |

**This README used to say "YOLOv8n would fit the budget at a cost in accuracy".
That was a guess, and it is wrong.** The nano model is 1.54x faster, not the 3x
assumed, and 567 s is still nearly double the 300 s budget. **No model here
reaches it.** The budget was written for hardware this is not, and swapping the
model does not rescue it.

What the smaller model actually costs is modest: 4.5% of Car AP, 9.5% of
Pedestrian AP, and 3.8 points of pedestrian recall ceiling. If the constraint
were throughput rather than a fixed budget, that is a defensible trade. It is
recorded here because "a smaller model would fix it" is the kind of sentence
that sounds like analysis and contains no information until somebody measures
it.

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

The budget is missed by the forward pass on a 15 W mobile CPU, not by the
pipeline around it, and **not by the choice of model**: the measurement above
shows YOLOv8n misses it too. The criterion was optimistic when it was written
and it is left standing, marked as missed, rather than quietly relaxed to
whatever the hardware happens to deliver.

## 📖 The analysis in full

The nine sections below are the detailed analysis: the sim-to-real comparison,
the error-type decomposition, the threshold argument, the SOTIF taxonomy and its
gate, metamorphic robustness, calibration sign, and the in-distribution check.
They live in [`docs/ANALYSIS.md`](docs/ANALYSIS.md) so that this file stays an
entry point.

- [M5: would the simulation have told you the same thing?](docs/ANALYSIS.md#m5-would-the-simulation-have-told-you-the-same-thing)
- [Was it missed, or just boxed badly?](docs/ANALYSIS.md#was-it-missed-or-just-boxed-badly)
- [And what kind of mistake was the wrong box?](docs/ANALYSIS.md#and-what-kind-of-mistake-was-the-wrong-box)
- [Beyond AP: what no threshold choice can buy](docs/ANALYSIS.md#beyond-ap-what-no-threshold-choice-can-buy)
- [Where would you set the threshold?](docs/ANALYSIS.md#where-would-you-set-the-threshold)
- [The SOTIF taxonomy, and the gate under it](docs/ANALYSIS.md#the-sotif-taxonomy-and-the-gate-under-it)
- [Metamorphic robustness: the same scene, degraded a stated amount](docs/ANALYSIS.md#metamorphic-robustness-the-same-scene-degraded-a-stated-amount)
- [Calibration, and why the sign matters more than the size](docs/ANALYSIS.md#calibration-and-why-the-sign-matters-more-than-the-size)
- [Is this frame the kind of thing we validated on?](docs/ANALYSIS.md#is-this-frame-the-kind-of-thing-we-validated-on)

## 🔮 Expected results, recorded before running

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

## 💡 What I learned

- **An aggregate metric is a place for failures to hide.** A detector at 0.68 mAP can
  be close to blind on occluded pedestrians past 40 metres, and the single number will
  never tell you. Slicing the evaluation is the whole point of the repository.

- **Validate your metric before you trust your result.** I checked the mAP
  implementation against pycocotools rather than assuming mine was right. If the ruler
  is wrong, every measurement taken with it is wrong in the same direction and nothing
  downstream will reveal it.

- **Writing down the expected result before running it is uncomfortable and useful.**
  It is the difference between a prediction and a rationalisation, and it is the
  cheapest defence against fitting the story to whatever came out.

- **Confidence calibration belongs in a detection benchmark.** A box is not just right
  or wrong. A detector that is confidently wrong is more dangerous in a vehicle than
  one that is uncertain and says so, and mAP alone cannot see that difference.

## 🔭 Future improvements

- **Extend the slices beyond distance and occlusion.** Truncation, lighting and object
  size each hide their own failures, and the harness is already shaped to take them.
- **Evaluate a second detector on the same slices.** One model tells you about that
  model; two tell you whether the slicing itself is finding something real.
- **Bring calibration into the headline, not a side section.** A reliability diagram
  belongs next to mAP, because a confidently wrong detection is the dangerous one.
- **Validate on a second dataset.** Everything here is KITTI, which is one camera, one
  city and one set of conditions.

## 📥 Getting the data

KITTI is not redistributed here. `data/` is gitignored; the committed 20-frame
fixture under `tests/fixtures/` is what the test suite runs against.

```
uv sync --group dev
uv run pytest
```

## 📄 Licence

Code under MIT, see [LICENSE](LICENSE).

The KITTI dataset is not included in this repository and is licensed separately
under CC BY-NC-SA 3.0 by Karlsruhe Institute of Technology and Toyota
Technological Institute at Chicago. The 20-frame fixture under `tests/fixtures/`
consists of annotation text files from that dataset, retained under the same
non-commercial terms for the purpose of testing this software.

---

Built by **Mo Kamel**, M.Eng. Mechatronic and Cyber-Physical Systems, Technische
Hochschule Deggendorf.
[Portfolio](https://mkamel7.github.io) · [LinkedIn](https://linkedin.com/in/mo-kamel7)
