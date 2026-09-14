# ADAS perception evaluation: the analysis in full

The detailed analysis behind the headline numbers in the
[README](../README.md). Every figure here traces to the same runs.

## M5: would the simulation have told you the same thing?

**No, and for a better reason than a bad correlation.**

**Virtual KITTI 2 contains no pedestrians.** Car 245 tracks, Van 22, Truck 6,
across all five scenes. This project's headline finding, that pedestrian
detection collapses beyond 30 m, **cannot be checked against the simulation at
all**, not mismeasured but untestable. A validation programme leaning on this
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

**Band by band the two agree closely.** Synthetic is equal or better in five of
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

## Metamorphic robustness: the same scene, degraded a stated amount

A second dataset changes the scene, the camera, the labelling policy and the class balance at once, so a drop in AP has four candidate causes. A perturbation changes exactly one thing by a stated amount and **leaves the ground truth identical**, so the curve is attributable. That is what makes these metamorphic relations rather than augmentations.

500 KITTI frames, `yolov8s`, IoU 0.5, worst drop relative to the unperturbed baseline (`scripts/sweep_robustness.py`, full curves in `outputs/robustness.md`):

| perturbation | at | Car | Pedestrian |
|---|---|---|---|
| blur | 4 px radius | -16.3% | **-17.4%** |
| contrast removed | 0.8 | -13.0% | **-17.6%** |
| JPEG | quality 10 | -8.4% | -11.7% |
| fog veil | 0.6 opacity | -8.0% | -7.2% |
| brightness | ±0.6 | **-0.2%** | -2.2% |

**Three findings.**

**Exposure is free and defocus is not.** Brightness at ±60% costs Car essentially nothing, which is a real result rather than a broken perturbation: the tests assert the image actually changed. A pipeline worrying about tunnel mouths and low sun is worrying about the wrong thing; one worrying about a dirty or misfocused lens is not.

**Pedestrians degrade faster than cars under every perturbation except fog.** The class that matters most for a braking decision is the more fragile one, and the gap widens with strength: at blur radius 2 the Car cost is 3.3% and the Pedestrian cost is 8.5%. A single aggregate mAP hides that completely.

**Nothing here falls off a cliff.** Every curve is gradual, so there is no threshold below which the detector stops working, and a degradation curve is the honest way to report that. A single number at one operating point would suggest a robustness the smooth decline does not contradict but also does not demonstrate.

**Read with three caveats, all of them stated in the code.** This is 500 frames, so the baselines here (Car 0.758, Pedestrian 0.443) are not the headline figures above, which come from all 7481. The fog is a **uniform veil, not depth-aware**, so it understates exactly the distance dependence that matters most for ADAS; `vkitti` is where depth-aware weather belongs. And Cyclist is mapping-limited to the point of meaninglessness here, so its column is omitted.

**Crop is deliberately not included.** It is a reasonable perturbation and it moves the boxes, so the ground truth would have to be transformed with it, which makes a bug in the box transform indistinguishable from a real drop. The whole point of this module is that nothing about the labels changes.

## Calibration, and why the sign matters more than the size

mAP asks how often the detector is right. **Calibration asks whether it knows how
often it is right**, and nothing else here measured that. A detector at 0.68 mAP
that reports 0.95 on every box it will get wrong is a worse engineering problem
than one reporting 0.4 on those boxes, because the second can be gated by a
threshold and the first cannot.

`src/ape/calibration.py` bins detections by confidence and reports what each band
actually delivered: a reliability diagram as data, plus expected calibration
error, maximum calibration error, and **overconfidence error**.

That last one is the point. **ECE is symmetric.** A detector claiming 0.4 while
being right 0.9 of the time scores exactly as badly as one claiming 0.9 while
being right 0.4 of the time, and those are not equally dangerous. The first is
timid and merely wastes performance; the second is **confidently wrong**, which
is the failure ISO 21448 exists for. A test constructs that exact pair and
asserts ECE cannot tell them apart while overconfidence error can.

**Slices are cut on the detection, not the ground truth**, which is the decision
here worth arguing with. Every other slice in this repository cuts on
ground-truth attributes: range, occlusion, truncation. Those exist only for
objects that are really there, so slicing calibration that way would silently
drop every false positive, and false positives are exactly where overconfidence
does its damage. Box height stands in for range. It is a weaker proxy than
KITTI's labelled distance and it is the only one a box corresponding to nothing
can have.

## Is this frame the kind of thing we validated on?

Every other measurement here asks how well the detector did on some data.
`src/ape/ood.py` asks the prior question: **is this data the data we validated
against.** A frame that is not is a triggering condition whether or not the
detector happened to get it right, which is the ISO 21448 case where nothing has
failed and the world is simply outside the design envelope.

It fits an operating envelope over six cheap image statistics and scores new
frames by Mahalanobis distance. **Mahalanobis rather than a z-score per feature
because the features covary**: a foggy frame is brighter *and* lower contrast
*and* has fewer edges together, and scoring each independently treats one
moderate joint excursion as three unremarkable ones. A test puts two probes the
same distance out on every individual feature, one along the correlation and one
across it, and asserts the second scores an order of magnitude higher.

**What it is not:** a learned OOD method. There is no network and nothing is
trained, consistent with the rest of this repository. It will notice fog, night,
blur, a blown exposure and compression artefacts. **It will not notice a
semantically novel object rendered at ordinary brightness and contrast**, and
that limit is the interesting half of the honesty, because it is exactly the
failure a statistics-only detector cannot see.

**An OOD score nobody has validated is a number, not evidence.** `agreement()`
measures whether high-scoring frames actually did worse, reporting an AUC that
sits at 0.5 for a score carrying no information. A score that cannot rank the
degraded frames first has not earned the right to gate anything. The output is
called `triggering_candidates` rather than triggering conditions on purpose: a
triggering condition is a scenario a person describes and reasons about, and
promoting a statistic straight into a safety artefact is the shortcut that name
refuses to take.

