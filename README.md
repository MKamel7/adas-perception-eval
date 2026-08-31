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

**All eight milestones are done. Milestone 2 MISSED its performance criterion**
and is recorded as missed rather than adjusted to fit the result. The criteria were
written before any code, so a result cannot be rationalised into a pass
afterwards, and that cuts both ways: M2's budget is missed and is recorded as
missed rather than adjusted.

## The result

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

## The showcase

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
| Report | a self-contained HTML file | a dashboard or a notebook. No server, no build step, the artifact is committed and readable |
| Uncertainty | frame-level bootstrap, 400 resamples | an object count. Ten objects in ten frames and ten in one frame are not equally informative and a count cannot tell them apart |
| CI | ruff, mypy strict, full suite at 90% coverage, traceability gate | nothing. CI cannot run the evaluation, since the data is 20 GB, and a workflow claiming otherwise would be lying |

`pycocotools` is a dependency of the **tests**, never of the pipeline. Nothing in
the measurement path may import it, or the validation would be circular.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Ingest and calibration | **done.** Labels parse; projection verified by a test that catches a deliberate sign flip, on a committed 20-frame fixture |
| M2 | Inference | **done, criterion missed.** Export is reproducible and the class mapping is data. 1500 frames took **13 minutes against a 5 minute budget**, see below |
| M3 | Metrics validated | **done.** Own AP agrees with `pycocotools` **exactly, to six decimal places**, against a required 0.001 |
| M4 | Slicing | **done.** Six dimensions, every one from a ground-truth attribute, all committed before the run |
| M5 | Sim-to-real | **done.** Six renders of Scene01, rank correlation 0.943 against real, and the finding below |
| M6 | Taxonomy | **done.** Six triggering conditions, three hazards, seven demonstrating tests, gated in both directions |
| M7 | Report | **done.** One command produces `outputs/report.html` |
| M8 | Demo and README | **done.** A 55 s showcase covering every finding, a 21 s distance scene, example frames chosen by rule, README and CV bullet below |

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

## M5: would the simulation have told you the same thing?

**No, and for a better reason than a bad correlation.**

**Virtual KITTI 2 contains no pedestrians.** Car 245 tracks, Van 22, Truck 6,
across all five scenes. This project's headline finding, that pedestrian
detection collapses beyond 30 m, **cannot be checked against the simulation at
all** — not mismeasured, untestable. A validation programme leaning on this
synthetic data would have had no way to see it. That is the sim-to-real result,
and it arrived before a single frame of inference.

For cars, which can be compared, the answer is more subtle and is the project's
own thesis turned on itself:

| distance | real AP | real share | synthetic AP | synthetic share |
|---|---|---|---|---|
| 0-10 m | 0.862 | 12.8% | 0.885 | 3.7% |
| 10-20 m | 0.804 | 22.9% | 0.975 | 6.9% |
| 20-30 m | 0.656 | 23.2% | 0.801 | 8.3% |
| 30-40 m | 0.518 | 17.4% | 0.557 | 9.8% |
| 40-50 m | 0.341 | 12.3% | 0.367 | 9.9% |
| >50 m | 0.172 | 11.4% | 0.098 | **61.5%** |
| **overall** | **0.754** | | **0.350** | |

**Band by band the two agree closely** — synthetic is equal or better in five of
six, and the difficulty ordering has a rank correlation of **0.943**, disagreeing
only on which of the two *easiest* bands is easiest. **In aggregate they look
completely different**, 0.754 against 0.350.

The gap is not behaviour, it is **composition**: 61.5% of the synthetic cars sit
beyond 50 m against 11.4% of the real ones. An aggregate sim-to-real comparison
would have concluded that the simulation behaves nothing like reality. It does;
it is simply populated differently. That is Simpson's paradox in a validation
pipeline, and it is the same argument this project makes about single numbers,
arriving uninvited in its own results.

### Weather and lighting, which KITTI has no examples of

| variant | Car AP | vs baseline |
|---|---|---|
| overcast | 0.360 | +0.010 |
| clone (baseline) | 0.350 | - |
| rain | 0.321 | -0.030 |
| morning | 0.318 | -0.033 |
| sunset | 0.299 | -0.051 |
| **fog** | **0.228** | **-0.122** |

Fog costs four times what rain does. Overcast is very slightly *better* than the
baseline render. These are synthetic weather effects and are labelled as such:
they are evidence about a renderer's fog, not about fog.

## Was it missed, or just boxed badly?

Everything above counts a miss as a miss. At IoU 0.5 that conflates two failures
with different fixes, and reporting them as one number sends an engineer to work
on the wrong half.

| class | AP@0.3 | AP@0.5 | **AP@0.7** | found | mislocated | unseen | of misses, a box problem |
|---|---|---|---|---|---|---|---|
| Car | 0.849 | 0.754 | **0.525** | 23511 | 3839 | 1392 | **73%** |
| Pedestrian | 0.596 | 0.506 | **0.266** | 3084 | 682 | 721 | 49% |

**Only 1392 of 28742 cars were not seen at all.** 73% of Car misses are boxes the
detector emitted on the object that did not overlap enough to score. The fix is
box regression, not recall, and that is the opposite of what the aggregate AP
suggests. Pedestrians split roughly evenly, so half of *that* problem really is
blindness, which corroborates the recall ceiling below.

For a safety argument the two are not equally alarming: a system that knows a
vehicle is roughly there can still brake for it. But a box 40% off feeds a wrong
position to whatever consumes it, so a tracker can place the vehicle in the
wrong lane with full confidence. The hazard shifts from a missed reaction to a
confidently wrong one.

**AP@0.7 is KITTI's own threshold for Car**, so that column is directly
comparable with the KITTI benchmark. This removes a limitation the README used
to carry.

## And what kind of mistake was the wrong box?

The section above explains the misses. Until now nothing explained the false
positives: every wrong box counted the same, so a detector that fires twice on
one pedestrian and a detector that invents pedestrians in empty road produced
the same number. AP cannot separate them either.

Counted over the whole curve, so these are every box the detector emits at any
confidence, not the ones a vehicle would act on:

| class | false positives | duplicate | misclassified | mislocalised | **hallucinated** |
|---|---|---|---|---|---|
| Car | 33752 | 266 (1%) | 81 (0%) | 5305 (16%) | **28100 (83%)** |
| Pedestrian | 10808 | 40 (0%) | 674 (6%) | 847 (8%) | **9247 (86%)** |

**Four fifths of the wrong boxes are on nothing at all**, and that is the
category a safety argument cares about most: it is the only one that makes a
vehicle brake for empty road, and the only one whose cause is invisible in the
ground truth. Duplicates are almost absent, so non-maximum suppression is not
the problem. Misclassification is a rounding error for Car and 6% for
Pedestrian, where the confusions are with the Cyclist and Car boxes a road
scene puts people next to.

The categories are defined in `src/ape/outcomes.py` and every one of them is
read off the same match the metric used, at the same threshold, in the same
order. Nothing here matches a second time.

## Beyond AP: what no threshold choice can buy

AP integrates over every operating point, which is a question no vehicle asks.
These are the ones it does ask.

| class | false-negative rate | recall at 90% precision | recall at 50% precision |
|---|---|---|---|
| Car | 18.2% | 64.7% | 80.9% |
| Pedestrian | 31.3% | **0.2%** | 60.7% |

**The pedestrian row is the finding.** An AP of 0.506 reads as a mediocre but
usable detector. It is not usable at high precision at all: demand 90%
precision and it returns two pedestrians in a thousand. There is no threshold
that buys both, and the aggregate hides that completely, which is the argument
for reporting more than one number per slice.

## Where would you set the threshold?

Average precision integrates over every confidence threshold at once. That is
right for comparing two detectors and useless for shipping one, because a
vehicle runs at a single threshold. Choosing it is the decision that turns an
evaluation into an engineering argument.

**Pedestrian recall tops out at 68.7% at ANY threshold.**

| target recall | threshold | precision | false alarms / frame |
|---|---|---|---|
| 50% | 0.522 | 0.660 | 0.15 |
| 70% | **unreachable** | | |
| 90% | **unreachable** | | |

That is not a tuning problem. Accepting *every* box the detector emits still
leaves a third of annotated pedestrians unreported. A limit no operating point
reaches is answered with a different sensor or a restricted operational domain,
not a different number, and telling those two situations apart is the most
consequential thing this evaluation does.

**Car reaches 81.8%, and the last stretch is expensive:**

| target recall | threshold | precision | false alarms / frame |
|---|---|---|---|
| 50% | 0.681 | 0.955 | 0.09 |
| 70% | 0.413 | 0.834 | 0.54 |
| 80% | 0.129 | 0.563 | **2.39** |
| 90% | **unreachable** | | |

A 1.6x gain in recall bought with a **26x** rise in phantom detections. At 2.39
false alarms per frame, a vehicle running ten frames a second reacts to
something imaginary twenty-two times a second.

**Both directions are hazards**, which is why the hazard analysis now has four
and not three: every one of the original set was about failing to react, and a
taxonomy containing only those is quietly arguing for a threshold of zero.
H-4 is unnecessary intervention, and phantom braking on a motorway is a
collision risk rather than a comfort complaint.

The rate is reported **per frame** rather than as precision because "one phantom
detection every four frames" is a quantity an integrator can hold against a
budget, and "precision 0.82" is not. No threshold is recommended: that depends
on the vehicle, the speed and the function, none of which are in this
repository.

## The SOTIF taxonomy, and the gate under it

`safety/triggering_conditions.yaml` records nine triggering conditions against
four hazards. SOTIF's device is four areas, known-safe, known-unsafe,
unknown-safe and **unknown-unsafe**, and the whole job is shrinking the last.
Each condition therefore states what it moved out of unknown-unsafe, not just
that a number was low.

| | Condition | Evidence |
|---|---|---|
| TC-01 | Pedestrian beyond ~30 m | AP 0.709 → 0.352 → 0.077 → 0.009 → 0.000 by range |
| TC-02 | Occlusion, the strongest predictor measured | Pedestrian 0.642 → 0.188 → 0.023 |
| TC-03 | Small apparent size, independent of range | Pedestrian under 40 px: 0.005 |
| TC-04 | Vehicles beyond 50 m | Car 0.900 → 0.172 |
| TC-05 | Truncation costs pedestrians, not cars | Car flat at 0.72-0.75; Pedestrian 0.479 → 0.103 |
| TC-06 | The class mapping cannot represent a cyclist | AP 0.006, a measurement artefact, not a detector limit |
| TC-07 | Pedestrian recall has a ceiling no threshold reaches | 68.7% at any operating point |
| TC-08 | Recall is bought with false alarms faster than linearly | Car 0.09 to 2.39 per frame for 50% to 80% |
| TC-09 | Vehicles are found but boxed loosely | AP 0.849 to 0.525 across IoU 0.3 to 0.7; 73% of misses are box problems |

**A negative result is recorded in the same file**: horizontal position in the
frame predicts nothing (Car 0.715 / 0.705 / 0.680 across thirds). A taxonomy
containing only the slices that worked is a fishing expedition with the evidence
removed.

**Every number in the taxonomy is checked against `outputs/results.json`, not a
sample of them.** That is not how it started: TC-01 originally claimed 0.529 at
10-20 m and 0.181 at 20-30 m, both typed from memory, and the test guarding it
verified two of its six figures which both happened to be right. Two fabricated
numbers sat in a safety argument behind a passing gate. The evidence is now
structured as `(slice, class, bin, ap)` so that "every entry" is something a
machine enumerates rather than something a person promises.

**The argument is gated in both directions.** A condition with no test fails the
build; a test naming a condition that does not exist fails it too, which is the
quieter failure and the one a single mistyped character causes. All three failure
modes have been watched failing, because a gate nobody has seen fail is an
assumption rather than a control.

The gate itself is `fih.gate`, **imported from the
[fault injection harness](https://github.com/MKamel7/fault-injection-harness)
rather than written a third time.** It was built there against ISO 26262
requirements and is used by the virtual production cell for PackML safety
requirements. This is its third safety argument and the first where the
requirements are perception insufficiencies rather than failures: the hazards
differ, the evidence differs, and the two ways an argument can have a hole do
not.

Example frames are rendered by `scripts/render_examples.py`, chosen **by rule**
(the frame with the most missed objects in that slice) rather than by eye, since
hand-picked frames would be illustrations of an argument instead of evidence for
it.

## CV bullet

> **ADAS perception evaluation pipeline:** slice-based detection metrics on
> KITTI with the mAP implementation validated against the reference COCO
> implementation to six decimal places, a six-condition triggering-condition
> taxonomy in ISO 21448 SOTIF vocabulary gated bidirectionally against its
> evidence, and the finding that a detector reported at 0.50 mAP for pedestrians
> scores 0.007 beyond 30 metres.

Say "SOTIF vocabulary", never "SOTIF compliant". Say "pretrained detector", never
imply training.

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

## Roadmap

- **Metamorphic robustness on the KITTI data already fetched** — brightness, blur, contrast, compression, crop, synthetic fog, reported as a degradation curve. Most of the domain-shift story at near-zero cost.
- **Confidence intervals on every slice cell**, not just the overall figures. The bootstrap already exists in `ape.uncertainty`. It is the difference between "night is worse" and "night is worse, and the sample supports saying so".
- **Calibration and OOD scoring** — reliability diagrams and expected calibration error per slice, then an OOD score feeding triggering-condition detection. When this detector says 0.9, how often is it right? A confidently wrong detector is a different safety problem from an uncertainly wrong one, and SOTIF cares far more about the first.

Not doing: **nuScenes, BDD100K or Waymo before the metamorphic curves exist** (large, licence-gated, and they answer a question the harness has not yet shown it can express). Not training a better detector either, which would make the numbers nicer and the point weaker.

## Licence

Code under MIT. KITTI is CC BY-NC-SA 3.0 and is not included.
