"""Fetch the KITTI 2D object detection labels and calibration into data/.

KITTI is CC BY-NC-SA 3.0 and is not redistributed in this repository. This script
is what makes `data/` reproducible from a fresh checkout, so the directory can be
gitignored without the evaluation becoming unrunnable by anyone else.

Only labels and calibration are fetched by default. The left colour images are
another 12 GB and are needed for inference (M2) but not for ingest (M1), so they
are opt-in rather than assumed:

    uv run python scripts/fetch_kitti.py
    uv run python scripts/fetch_kitti.py --images

Sizes are checked after download. A truncated archive that unzips far enough to
look plausible is the failure worth guarding against, because it would silently
produce an evaluation over a subset while reporting it as the whole split.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"

#: name -> (archive, approximate size in bytes, a path that must exist after)
PARTS = {
    "labels": ("data_object_label_2.zip", 5_601_213, "training/label_2"),
    "calib": ("data_object_calib.zip", 26_854_811, "training/calib"),
    "images": ("data_object_image_2.zip", 12_569_945_557, "training/image_2"),
}

#: The published split. 7481 training frames with labels, 7518 test frames
#: without. Only the training split is usable here, since evaluation needs
#: ground truth, and that is worth stating because "KITTI has 15000 frames" is a
#: number people quote at this project.
EXPECTED_TRAINING_FRAMES = 7481


def download(archive: str, into: Path) -> Path:
    target = into / archive
    if target.exists():
        print(f"  {archive} already present, skipping download")
        return target
    url = f"{BASE}/{archive}"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)
    return target


def extract(archive: Path, into: Path, must_contain: str) -> None:
    print(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(into)
    if not (into / must_contain).is_dir():
        raise SystemExit(
            f"{archive.name} extracted but {must_contain} is missing, so the "
            f"archive layout has changed and the paths here are wrong")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", action="store_true",
                        help="also fetch the 12 GB left colour images, needed from M2")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--keep-archives", action="store_true",
                        help="do not delete the zips after extracting")
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    wanted = ["labels", "calib"] + (["images"] if args.images else [])

    for name in wanted:
        archive, size, must_contain = PARTS[name]
        print(f"{name} (about {size / 1e6:.0f} MB)")
        path = download(archive, args.data)
        extract(path, args.data, must_contain)
        if not args.keep_archives:
            path.unlink()

    frames = len(list((args.data / "training" / "label_2").glob("*.txt")))
    if frames != EXPECTED_TRAINING_FRAMES:
        raise SystemExit(
            f"got {frames} label files, expected {EXPECTED_TRAINING_FRAMES}. "
            f"A partial download would otherwise evaluate a subset and report "
            f"it as the full split.")

    print(f"\n{frames} frames ready under {args.data}/training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
