"""Export a pretrained YOLO model to ONNX, reproducibly.

    uv run --extra export python scripts/export_model.py

WHY THIS IS A SCRIPT AND NOT A COMMITTED .onnx FILE. A binary blob in the
repository is a model nobody can review, whose provenance is a claim in a README
rather than something a reader can check. This script names the weights, the
opset and the input size, so anyone can regenerate the exact file and compare.

WHY EXPORT AT ALL, since ultralytics can run the model directly. ONNX is what
ships to embedded automotive targets, and it is several times faster on a CPU
than the PyTorch path. Keeping the export in the repository also keeps the
heavyweight dependency OUT of the evaluation: `ultralytics` and `torch` are
needed here and nowhere else, so the pipeline that produces the numbers depends
on onnxruntime alone.

The weights are COCO-pretrained and nothing is trained here. That is the point
of the project, and it is stated in the README rather than implied.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "models"

#: Opset 12 is old enough to be accepted by the embedded runtimes this format
#: exists to reach, and new enough for everything the model needs.
OPSET = 12

#: 640 is what the model was trained at. Exporting at another size silently
#: changes the accuracy being measured, so it is named rather than defaulted.
IMAGE_SIZE = 640


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolov8s.pt",
                        help="pretrained COCO weights (default: yolov8s.pt)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE)
    args = parser.parse_args()

    from ultralytics import YOLO

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.weights}")
    model = YOLO(args.weights)

    print(f"exporting to ONNX, opset {OPSET}, {args.imgsz}x{args.imgsz}")
    produced = Path(model.export(format="onnx", opset=OPSET, imgsz=args.imgsz,
                                 simplify=False, dynamic=False))

    target = args.out / f"{Path(args.weights).stem}.onnx"
    if produced.resolve() != target.resolve():
        shutil.move(str(produced), target)

    size = target.stat().st_size
    print(f"\nwrote {target} ({size / 1e6:.1f} MB)")
    print("The evaluation needs only onnxruntime from here; ultralytics and "
          "torch are export-time dependencies and nothing else imports them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
