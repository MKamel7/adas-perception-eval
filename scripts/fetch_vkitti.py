"""Fetch Virtual KITTI 2, the synthetic half of the sim-to-real comparison.

    uv run python scripts/fetch_vkitti.py

WHY THIS DATASET AND NOT A DIFFERENT SYNTHETIC ONE. Virtual KITTI 2 re-renders
KITTI's own sequences, with the same camera geometry, in weather and lighting
variants KITTI does not contain: fog, rain, morning, sunset, overcast. That
shared content is the entire point. Comparing a detector on KITTI against a
detector on some unrelated synthetic dataset measures the gap between two
different worlds; comparing it on the SAME scenes re-rendered measures the gap
between real and synthetic, which is the question worth asking.

WHAT THE COMPARISON CAN AND CANNOT SHOW. Absolute scores will not match and are
not expected to: synthetic imagery is cleaner and the annotation is perfect. The
question is whether the SHAPE of the degradation rhymes, whether the things that
are hard in simulation are the things that are hard in reality. If they do,
simulation-based validation has some predictive value here. If they do not, that
is the more interesting result and it gets reported just as loudly.

Resumable, and refuses to extract a short archive, for the same reason
fetch_kitti.py does: an interrupted multi-gigabyte transfer otherwise leaves a
plausible-looking file that unzips far enough to produce an evaluation over a
subset nobody chose.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3"

#: name -> (archive, expected bytes, a path that must exist afterwards)
PARTS = {
    "labels": ("vkitti_2.0.3_textgt.tar.gz", None, "Scene01"),
    "rgb": ("vkitti_2.0.3_rgb.tar", 7_542_648_320, "vkitti_2.0.3_rgb"),
}


def size_of(url: str) -> int | None:
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except OSError:
        return None


def download(archive: str, into: Path) -> Path:
    """Fetch or resume, and never return a short file."""
    url = f"{BASE}/{archive}"
    target = into / archive
    expected = size_of(url)
    have = target.stat().st_size if target.exists() else 0

    if expected and have == expected:
        print(f"  {archive} already complete")
        return target
    if expected and have > expected:
        raise SystemExit(f"{target} is larger than the server's copy; delete it")

    request = urllib.request.Request(url)
    mode = "wb"
    if have and expected:
        print(f"  resuming at {have / 1e9:.2f} of {expected / 1e9:.2f} GB", flush=True)
        request.add_header("Range", f"bytes={have}-")
        mode = "ab"
    else:
        print(f"  downloading {url}"
              + (f" ({expected / 1e9:.1f} GB)" if expected else ""), flush=True)

    with urllib.request.urlopen(request) as response:
        if have and response.status != 206:
            # The server ignored the range. Appending a second copy of the whole
            # file to a partial one produces a corrupt archive of exactly the
            # right-looking size, which is the worst possible outcome.
            print("  no resume support, restarting", flush=True)
            mode, have = "wb", 0
        with target.open(mode) as handle:
            shutil.copyfileobj(response, handle)

    got = target.stat().st_size
    if expected and got != expected:
        raise SystemExit(
            f"{archive} is {got} bytes, expected {expected}. Run again to "
            f"resume rather than extracting a truncated archive.")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/vkitti")
    parser.add_argument("--only", choices=sorted(PARTS))
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()

    args.data.mkdir(parents=True, exist_ok=True)
    wanted = [args.only] if args.only else ["labels", "rgb"]

    for name in wanted:
        archive, _, must_contain = PARTS[name]
        print(f"{name}", flush=True)
        path = download(archive, args.data)
        print(f"  extracting {archive}", flush=True)
        with tarfile.open(path) as handle:
            handle.extractall(args.data, filter="data")
        if not any(args.data.glob(f"**/{must_contain}*")):
            raise SystemExit(
                f"{archive} extracted but nothing matching {must_contain} "
                f"appeared; the archive layout has changed")
        if not args.keep_archives:
            path.unlink()

    print(f"\nready under {args.data}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
