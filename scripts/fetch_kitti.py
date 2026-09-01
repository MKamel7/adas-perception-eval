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


def download(archive: str, into: Path, expected: int) -> Path:
    """Fetch, or resume a partial fetch, and never hand back a short file.

    The first version of this skipped the download whenever the file existed,
    which is the exact failure the module docstring warns about: an interrupted
    12 GB transfer leaves a plausible-looking zip, the next run trusts it, and
    the evaluation quietly proceeds over whatever fraction arrived. That is not
    hypothetical either, it happened on the first attempt here.

    Resuming rather than restarting because at this size a dropped connection is
    routine, and re-fetching 3 GB that are already on disk to get the next 9 is
    a bad trade.
    """
    target = into / archive
    have = target.stat().st_size if target.exists() else 0

    if have == expected:
        print(f"  {archive} already complete, skipping")
        return target
    if have > expected:
        raise SystemExit(
            f"{target} is {have} bytes but should be {expected}. Refusing to "
            f"guess what it is; delete it and run again.")

    url = f"{BASE}/{archive}"
    request = urllib.request.Request(url)
    mode = "wb"
    if have:
        print(f"  resuming {archive} at {have / 1e9:.2f} GB of {expected / 1e9:.2f} GB")
        request.add_header("Range", f"bytes={have}-")
        mode = "ab"
    else:
        print(f"  downloading {url}")

    with urllib.request.urlopen(request) as response:
        if have and response.status != 206:
            # The server ignored the range and is sending the whole file from
            # the start. Appending that to what is already there would produce
            # a corrupt archive of exactly the right-looking size.
            print("  server does not support resuming, restarting the download")
            mode, have = "wb", 0
        with target.open(mode) as out:
            shutil.copyfileobj(response, out)

    got = target.stat().st_size
    if got != expected:
        raise SystemExit(
            f"{archive} is {got} bytes, expected {expected}. The transfer was "
            f"cut short; run again to resume rather than extracting this.")
    return target


def extract(archive: Path, into: Path, must_contain: str) -> None:
    print(f"  extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        # Every member is checked to land inside `into` before anything is
        # written. ZipFile has no equivalent of tarfile's filter="data", which
        # is what fetch_vkitti.py uses, so the check is written out here. An
        # entry named "../../etc/thing" or an absolute path would otherwise
        # extract outside the data directory; these archives come from a fixed
        # official URL, so this is a guard against the URL or the host
        # changing, not against KITTI.
        root = into.resolve()
        for member in zf.infolist():
            destination = (root / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise SystemExit(
                    f"{archive.name} contains {member.filename!r}, which would "
                    f"extract outside {into}. Refusing to unpack it.")
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
    parser.add_argument("--only", choices=sorted(PARTS),
                        help="fetch just one part, for resuming a long transfer "
                             "without re-extracting the small ones")
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    if args.only:
        wanted = [args.only]
    else:
        wanted = ["labels", "calib"] + (["images"] if args.images else [])

    for name in wanted:
        archive, size, must_contain = PARTS[name]
        print(f"{name} (about {size / 1e6:.0f} MB)", flush=True)
        path = download(archive, args.data, size)
        extract(path, args.data, must_contain)
        if not args.keep_archives:
            path.unlink()

    labels = args.data / "training" / "label_2"
    if not labels.is_dir():
        print(f"\n{args.only or 'requested parts'} ready under {args.data}/training")
        return 0

    frames = len(list(labels.glob("*.txt")))
    if frames != EXPECTED_TRAINING_FRAMES:
        raise SystemExit(
            f"got {frames} label files, expected {EXPECTED_TRAINING_FRAMES}. "
            f"A partial download would otherwise evaluate a subset and report "
            f"it as the full split.")

    print(f"\n{frames} frames ready under {args.data}/training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
