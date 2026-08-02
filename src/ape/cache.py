"""Read and write the detection cache.

Inference costs minutes; everything after it costs seconds. Caching the
detections means a change to the METRIC can be re-run against byte-identical
detector output, so any difference in the result is attributable to the change
rather than to a different run of the model. Without that, tuning the evaluation
and tuning the detector look the same from the outside.

JSONL rather than a pickle or a database: it is greppable, diffable, and a
truncated file fails on the line it was cut at instead of loading as a shorter
evaluation nobody notices.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ape.records import Box2D, Detection


@dataclass(frozen=True)
class Header:
    """What produced these detections. Without it the numbers are unattributable."""

    model: str
    frames: int
    score_threshold: float
    first_frame: str
    last_frame: str


class CacheError(ValueError):
    """A cache that cannot be trusted, as opposed to one that is absent."""


def load_detections(path: Path) -> tuple[Header, dict[str, list[Detection]]]:
    """Every cached detection, keyed by frame.

    The header is required. A detection file with no record of which model or
    which split produced it can still be evaluated, and the resulting number
    cannot be compared with anything or reproduced, which makes it worse than
    no number at all.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise CacheError(f"{path} is empty")

    first = json.loads(lines[0])
    if first.get("kind") != "header":
        raise CacheError(
            f"{path} has no header, so the model and split that produced it "
            f"are unknown and its numbers cannot be attributed")
    header = Header(model=first["model"], frames=first["frames"],
                    score_threshold=first["score_threshold"],
                    first_frame=first["first_frame"],
                    last_frame=first["last_frame"])

    by_frame: dict[str, list[Detection]] = defaultdict(list)
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as bad:
            raise CacheError(
                f"{path}:{number} is not valid JSON. A truncated cache must "
                f"fail here rather than load as a shorter evaluation.") from bad
        by_frame[row["frame_id"]].append(Detection(
            frame_id=row["frame_id"], label=row["label"], score=row["score"],
            box=Box2D(*row["box"])))
    return header, dict(by_frame)
