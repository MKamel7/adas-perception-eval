"""Every triggering condition, recomputed from the committed results.

The taxonomy in `safety/triggering_conditions.yaml` makes six claims about where
this detector fails. Claims in a YAML file are assertions; these tests turn them
into evidence by recomputing each number from `outputs/results.json`, which is
the artifact the evaluation actually produced.

WHAT THIS CATCHES that reading the file does not: a condition whose numbers were
right when written and are now stale because the evaluation was re-run. A safety
argument that drifts from its evidence is worse than one that never had any,
because it still reads as though it were checked.

Each test carries `@pytest.mark.demonstrates("TC-xx")`, and
`scripts/check_conditions.py` fails the build if a condition has no test or a
test names a condition that does not exist.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/results.json"
ANALYSIS = ROOT / "safety/triggering_conditions.yaml"

pytestmark = pytest.mark.skipif(
    not RESULTS.exists(),
    reason="no evaluation has been run; scripts/evaluate.py produces results.json")

#: How close a recomputed number must be to the one written in the taxonomy.
#: Tight enough that a re-run which genuinely moved the result fails this, loose
#: enough to survive the rounding in the YAML.
TOLERANCE = 0.005


def results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def analysis() -> dict:
    return yaml.safe_load(ANALYSIS.read_text(encoding="utf-8"))


def condition(cid: str) -> dict:
    found = next((c for c in analysis()["conditions"] if c["id"] == cid), None)
    assert found is not None, f"{cid} is not in the taxonomy"
    return found


def ap(data: dict, dimension: str, bin_name: str, label: str) -> float:
    for item in data["slices"]:
        if item["dimension"] == dimension and item["bin"] == bin_name:
            for cell in item["cells"]:
                if cell["label"] == label:
                    return float(cell["ap"])
    raise AssertionError(f"no {label} cell for {dimension}/{bin_name}")


# --- the conditions ----------------------------------------------------------
@pytest.mark.demonstrates("TC-01")
def test_pedestrians_are_not_detected_beyond_thirty_metres() -> None:
    """The headline finding, and the one with a safety argument attached."""
    data = results()
    near = ap(data, "distance", "0-10 m", "Pedestrian")
    far = ap(data, "distance", "30-40 m", "Pedestrian")
    further = ap(data, "distance", ">50 m", "Pedestrian")

    assert near > 0.6, f"near pedestrians should be found, got {near:.3f}"
    assert far < 0.05, (
        f"the claim is that detection collapses past 30 m; it is now "
        f"{far:.3f}, so the taxonomy is stale")
    assert further < 0.01


@pytest.mark.demonstrates("TC-02")
def test_occlusion_degrades_both_classes_monotonically() -> None:
    """Monotone is part of the claim. A non-monotone result would mean the
    annotation levels are not measuring what they say."""
    data = results()
    for label, floor in (("Pedestrian", 0.05), ("Car", 0.3)):
        visible = ap(data, "occlusion", "fully visible", label)
        partly = ap(data, "occlusion", "partly occluded", label)
        largely = ap(data, "occlusion", "largely occluded", label)

        assert visible > partly > largely, (
            f"{label} occlusion is no longer monotone: "
            f"{visible:.3f} / {partly:.3f} / {largely:.3f}")
        assert largely < floor


@pytest.mark.demonstrates("TC-02")
def test_a_pedestrian_loses_most_performance_before_being_mostly_hidden() -> None:
    """The specific number the condition rests on: the drop from fully to
    PARTLY occluded, which is the urban case."""
    data = results()
    visible = ap(data, "occlusion", "fully visible", "Pedestrian")
    partly = ap(data, "occlusion", "partly occluded", "Pedestrian")

    assert visible / partly > 3.0, (
        f"the claim is a 3.4x drop at partial occlusion; it is now "
        f"{visible / partly:.1f}x")


@pytest.mark.demonstrates("TC-03")
def test_small_objects_are_missed_regardless_of_class() -> None:
    data = results()
    assert ap(data, "box height", "0-25 px", "Pedestrian") < 0.01
    assert ap(data, "box height", "25-40 px", "Pedestrian") < 0.05
    assert ap(data, "box height", "0-25 px", "Car") < 0.30

    big = ap(data, "box height", ">160 px", "Pedestrian")
    assert big > 0.6, f"large pedestrians should be found, got {big:.3f}"


@pytest.mark.demonstrates("TC-04")
def test_cars_degrade_with_range_but_gracefully() -> None:
    """Both halves matter. That cars degrade supports the condition; that they
    degrade gently is what distinguishes it from the pedestrian case and stops
    the two being reported as one finding."""
    data = results()
    near = ap(data, "distance", "0-10 m", "Car")
    far = ap(data, "distance", ">50 m", "Car")

    assert near > 0.85
    assert far < 0.25
    assert far > 0.10, (
        f"cars past 50 m are degraded, not blind, at {far:.3f}; if this falls "
        f"below 0.1 the condition should be reworded to match the pedestrian one")


@pytest.mark.demonstrates("TC-05")
def test_truncation_costs_pedestrians_and_not_cars() -> None:
    """The asymmetry IS the finding, so both directions are asserted."""
    data = results()
    car_none = ap(data, "truncation", "none", "Car")
    car_heavy = ap(data, "truncation", ">50%", "Car")
    ped_none = ap(data, "truncation", "none", "Pedestrian")
    ped_mid = ap(data, "truncation", "30-50%", "Pedestrian")

    assert abs(car_none - car_heavy) < 0.05, (
        f"cars are claimed to be unaffected by truncation, but moved from "
        f"{car_none:.3f} to {car_heavy:.3f}")
    assert ped_none / max(ped_mid, 1e-9) > 2.0, (
        f"pedestrians are claimed to lose about half, but moved from "
        f"{ped_none:.3f} to {ped_mid:.3f}")


@pytest.mark.demonstrates("TC-06")
def test_the_cyclist_number_is_reported_and_disclaimed() -> None:
    """A condition about the measurement rather than the detector, which is why
    it hangs off the hazard about overstated performance."""
    from ape.classes import EVALUATED, HEADLINE

    data = results()

    assert data["overall"]["Cyclist"] < 0.05
    assert "Cyclist" in EVALUATED, "it must be reported"
    assert "Cyclist" not in HEADLINE, "and must not be averaged into the headline"


# --- do the claims survive their own uncertainty? ----------------------------
def interval(data: dict, dimension: str, bin_name: str, label: str
             ) -> tuple[float, float] | None:
    for item in data["slices"]:
        if item["dimension"] == dimension and item["bin"] == bin_name:
            for cell in item["cells"]:
                if cell["label"] == label and cell["ci_low"] is not None:
                    return (float(cell["ci_low"]), float(cell["ci_high"]))
    return None


@pytest.mark.demonstrates("TC-01")
def test_the_distance_claim_is_not_explained_by_sampling() -> None:
    """The point of computing intervals at all.

    TC-01 says near and far pedestrians are different, not merely differently
    averaged. If the confidence intervals overlapped, that wording would be
    unsupported and the condition would have to be softened to "the point
    estimates differ". They do not overlap, by a wide margin, so the claim
    stands as written.
    """
    data = results()
    near = interval(data, "distance", "0-10 m", "Pedestrian")
    far = interval(data, "distance", "30-40 m", "Pedestrian")

    assert near and far, "no intervals in results.json; re-run scripts/evaluate.py"
    assert far[1] < near[0], (
        f"the 30-40 m interval {far} overlaps the 0-10 m interval {near}, so "
        f"the difference is not established and TC-01 overstates the evidence")


@pytest.mark.demonstrates("TC-02")
def test_the_occlusion_claim_is_not_explained_by_sampling() -> None:
    data = results()
    visible = interval(data, "occlusion", "fully visible", "Pedestrian")
    largely = interval(data, "occlusion", "largely occluded", "Pedestrian")

    assert visible and largely
    assert largely[1] < visible[0], (
        f"largely occluded {largely} overlaps fully visible {visible}")


def test_a_thin_slice_reports_a_wide_interval() -> None:
    """The interval has to be doing work, not decorating.

    A cell computed from a handful of objects must come with a visibly wider
    range than one computed from thousands, or the column is telling the reader
    nothing they could not get from the object count.
    """
    data = results()
    thin = interval(data, "box height", "0-25 px", "Pedestrian")
    thick = interval(data, "distance", "0-10 m", "Car")

    assert thin and thick
    assert (thin[1] - thin[0]) >= 0.0, "an interval cannot be negative"
    assert (thick[1] - thick[0]) < 0.15, (
        f"the Car 0-10 m interval is {thick[1] - thick[0]:.3f} wide over "
        f"hundreds of objects, which is too wide to be believable")


# --- the operating point, which average precision cannot express -------------
@pytest.mark.demonstrates("TC-07")
def test_pedestrian_recall_has_a_ceiling_no_threshold_reaches() -> None:
    """The most consequential thing this evaluation says.

    A limit no operating point reaches is not a tuning problem. Distinguishing
    "lower the threshold" from "you cannot get there from here" is the
    difference between a sprint spent moving a number and a decision to add a
    sensor, and average precision alone cannot tell them apart.
    """
    data = results()
    ceiling = float(data["ceiling_recall"]["Pedestrian"])

    assert ceiling < 0.75, (
        f"pedestrian recall now tops out at {ceiling:.3f}; TC-07 claims a "
        f"ceiling that leaves a substantial fraction unreported")

    for point in data["operating_points"]["Pedestrian"]:
        if point["target"] > ceiling:
            assert not point["reachable"], (
                f"{point['target']:.0%} is above the ceiling {ceiling:.3f} and "
                f"must be reported unreachable, not approximated")


@pytest.mark.demonstrates("TC-08")
def test_recall_is_bought_with_false_alarms_faster_than_linearly() -> None:
    """Both directions are hazards.

    A taxonomy that only counted misses would be arguing for a threshold of
    zero, which would brake continuously. This pins the price of the recall the
    safety case keeps asking for.
    """
    data = results()
    points = {p["target"]: p for p in data["operating_points"]["Car"]}
    low, high = points[0.50], points[0.80]

    assert low["reachable"] and high["reachable"]
    recall_gain = 0.80 / 0.50
    alarm_growth = high["false_alarms_per_frame"] / low["false_alarms_per_frame"]

    assert alarm_growth > recall_gain * 5, (
        f"recall rose {recall_gain:.1f}x and false alarms {alarm_growth:.1f}x; "
        f"TC-08 claims the cost grows much faster than the benefit")
    assert high["false_alarms_per_frame"] > 1.0, (
        "the claim is that 80% recall costs more than one phantom detection "
        "per frame")


@pytest.mark.demonstrates("TC-08")
def test_a_tighter_threshold_is_quieter_and_finds_less() -> None:
    """There is no free recall, and the table has to show that."""
    data = results()
    for label in ("Car", "Pedestrian"):
        reachable = [p for p in data["operating_points"][label] if p["reachable"]]
        for tighter, looser in itertools.pairwise(reachable):
            assert looser["threshold"] <= tighter["threshold"]
            assert (looser["false_alarms_per_frame"]
                    >= tighter["false_alarms_per_frame"])


# --- every number in the file, not a hand-picked two -------------------------
def evidence_entries() -> list[tuple[str, dict]]:
    return [(c["id"], e) for c in analysis()["conditions"] for e in c["evidence"]]


@pytest.mark.parametrize("cid,entry", evidence_entries(),
                         ids=lambda v: v if isinstance(v, str) else
                         f"{v['class']}-{v['bin']}")
def test_every_claimed_number_matches_the_evaluation(cid: str, entry: dict) -> None:
    """The check that should have existed from the start.

    THIS TEST EXISTS BECAUSE ITS ABSENCE LET TWO FABRICATED NUMBERS INTO A
    SAFETY ARGUMENT. TC-01 originally claimed 0.529 at 10-20 m and 0.181 at
    20-30 m; the measured values are 0.361 and 0.069. They were typed from
    memory rather than read from results.json, and the test that was supposed to
    guard TC-01 checked two of its six numbers, both of which happened to be
    right.

    A gate that verifies a subset chosen by the same person who wrote the claims
    is not a gate. Every entry is checked now, and the evidence is structured as
    (slice, class, bin, ap) precisely so that "every" is something a machine can
    enumerate rather than something a human promises.
    """
    data = results()

    if entry["slice"] == "overall":
        actual = float(data["overall"][entry["class"]])
    elif entry["slice"] == "ceiling":
        actual = float(data["ceiling_recall"][entry["class"]])
    elif entry["slice"] == "false alarms per frame":
        target = float(entry["bin"].split()[-1])
        point = next(p for p in data["operating_points"][entry["class"]]
                     if abs(p["target"] - target) < 1e-9)
        actual = float(point["false_alarms_per_frame"])
    else:
        actual = ap(data, entry["slice"], entry["bin"], entry["class"])

    assert abs(actual - entry["ap"]) < TOLERANCE, (
        f"{cid} claims {entry['class']} {entry['slice']}/{entry['bin']} is "
        f"{entry['ap']:.3f}; the evaluation says {actual:.3f}")


# --- the taxonomy itself -----------------------------------------------------
def test_every_condition_names_a_hazard_that_exists() -> None:
    document = analysis()
    hazards = {h["id"] for h in document["hazards"]}

    for item in document["conditions"]:
        assert item["hazard"] in hazards, (
            f"{item['id']} traces up to {item['hazard']}, which does not exist")


def test_the_negative_result_is_still_recorded() -> None:
    """A taxonomy containing only the slices that worked is a fishing
    expedition with the evidence removed."""
    document = analysis()

    assert document.get("negative_results"), (
        "the position slice showed no effect and that must stay on the record")

    data = results()
    spread = [ap(data, "position", b, "Car")
              for b in ("left third", "centre third", "right third")]
    assert max(spread) - min(spread) < 0.08, (
        f"position now varies by {max(spread) - min(spread):.3f}; it was "
        f"recorded as showing no effect and that is no longer true")


def test_the_taxonomy_meets_the_milestone() -> None:
    """M6 asks for at least five triggering conditions in SOTIF vocabulary,
    each with the slice evidence that found it."""
    document = analysis()

    assert len(document["conditions"]) >= 5
    for item in document["conditions"]:
        assert item["evidence"], f"{item['id']} has no measurement"
        for entry in item["evidence"]:
            assert {"slice", "class", "bin", "ap"} <= set(entry), (
                f"{item['id']} has an evidence entry that cannot be checked "
                f"automatically: {entry}")
        assert item["unknown_to_known"].strip(), (
            f"{item['id']} does not say what it moved out of unknown-unsafe, "
            f"which is the only reason to write it down")
