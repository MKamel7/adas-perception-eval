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

Milestone 1 of 8. Ingest and calibration are in and tested; inference, metrics,
slicing and the report are not yet written. The
[milestone table](#milestones) says what each one has to prove before it counts as
done, and those criteria were written before any code, so a result cannot be
rationalised into a pass afterwards.

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
| M1 | Ingest and calibration | Labels parse, projection verified by a test that catches a deliberate sign flip, on a committed 20-frame fixture |
| M2 | Inference | ONNX export reproducible, 1500 frames in under 5 minutes on 12 CPU threads, COCO to KITTI class mapping written down as data |
| M3 | Metrics validated | Own mAP agrees with `pycocotools` **to within 0.001** on identical inputs. The gate the project lives or dies on |
| M4 | Slicing | At least 6 slice dimensions, each traceable to a ground-truth attribute, none picked after seeing results |
| M5 | Sim-to-real | Same pipeline on Virtual KITTI 2, degradation curves compared, agreement or disagreement reported either way |
| M6 | Taxonomy | At least 5 triggering conditions in SOTIF vocabulary, each with example frames and the slice evidence that found it |
| M7 | Report | One command produces the HTML from a fresh checkout plus data |
| M8 | Demo | A short scene showing the pipeline and the sliced result |

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
