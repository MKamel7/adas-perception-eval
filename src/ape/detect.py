"""Run an exported YOLO model over frames and emit canonical `Detection`s.

ONNX Runtime rather than PyTorch, deliberately. ONNX is the format that actually
ships to embedded automotive targets, it is several times faster on a CPU, and
the export step is the industry-relevant part that most portfolio projects skip
entirely. Nothing downstream of this module knows which detector produced its
input, which is what lets the same evaluation run against a different model
without touching the measurement path.

THE PART THAT SILENTLY GOES WRONG is not the inference, it is the coordinates.
The model sees a letterboxed square; the ground truth is in original image
pixels. Every box has to come back through that transform exactly, and an error
here does not raise: it shifts every detection by a few pixels, lowers IoU
slightly, and reports a detector that is a little worse than it really is. The
inverse transform is therefore written once, in `Letterbox`, and tested by
round-tripping rather than by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ape.classes import to_kitti
from ape.records import Box2D, Detection

if TYPE_CHECKING:
    import numpy as np

#: COCO class names in the order the model emits them.
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class Letterbox:
    """The resize-and-pad that got an image into the model's square input.

    Kept as a value rather than recomputed, because the forward and inverse
    transforms must agree exactly and the cheapest way to guarantee that is for
    only one of them to exist.
    """

    scale: float
    pad_x: float
    pad_y: float

    @classmethod
    def fit(cls, width: int, height: int, size: int) -> Letterbox:
        scale = min(size / width, size / height)
        return cls(scale=scale,
                   pad_x=(size - width * scale) / 2,
                   pad_y=(size - height * scale) / 2)

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        """Model coordinates back to original image pixels."""
        return (x - self.pad_x) / self.scale, (y - self.pad_y) / self.scale


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression, per class, vectorised.

    Written out rather than pulled from a library for the same reason IoU is:
    it is short, it is in the measurement path, and a subtly wrong version
    produces plausible numbers instead of an error.

    I CLAIMED THIS WAS THE BOTTLENECK AND I WAS WRONG, which is worth leaving
    written down. The first version built two `Box2D` objects per comparison in
    a Python loop, and at the 0.05 confidence threshold average precision needs
    it had roughly twenty times more candidates than a demo would. It looked
    like the obvious culprit for a run that took 844 seconds against a budget
    of 300, so it was rewritten. The run then took 799 seconds.

    Profiling afterwards, which should have come first, put the time where it
    actually was, per frame:

      ONNX forward   397 ms    83%
      preprocess      35 ms
      image load      34 ms
      postprocess     12 ms     2.4%   <- what this rewrite addressed

    The vectorisation is kept because it is strictly better and the arithmetic
    is identical. But the budget is missed by the model's forward pass on a 15 W
    mobile CPU, not by this function, and the README says so rather than
    implying the code was tuned into shape.
    """
    import numpy as np

    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: list[int] = []
    while order.size:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[best] + areas[rest] - inter
        # Guard the degenerate case rather than letting it become a warning and
        # a silent NaN that compares false against every threshold.
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
        order = rest[iou <= threshold]
    return keep


class Detector:
    """An exported YOLO model behind the canonical `Detection` interface."""

    def __init__(self, model: Path, *, score_threshold: float = 0.25,
                 iou_threshold: float = 0.45, threads: int = 0) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # A square input is assumed by the letterbox. Assert it rather than
        # discover it as a coordinate bug later.
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640
        if isinstance(shape[3], int) and shape[3] != self.size:
            raise ValueError(
                f"model input is {shape[2]}x{shape[3]}, not square; the "
                f"letterbox transform assumes a square input")
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def _preprocess(self, image: Any) -> tuple[np.ndarray, Letterbox]:
        import numpy as np
        from PIL import Image

        width, height = image.size
        box = Letterbox.fit(width, height, self.size)
        resized = image.resize(
            (max(1, round(width * box.scale)), max(1, round(height * box.scale))),
            Image.Resampling.BILINEAR)
        # 114 grey is what the reference implementation pads with. The value
        # matters: pad with black and the model sees a hard edge that is not in
        # the scene.
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(resized, (round(box.pad_x), round(box.pad_y)))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None]), box

    def detect(self, image_path: Path, frame_id: str) -> list[Detection]:
        """Every scored detection in one frame, in original image pixels."""
        from PIL import Image

        with Image.open(image_path) as handle:
            return self.detect_image(handle.convert("RGB"), frame_id)

    def detect_image(self, image: Any, frame_id: str) -> list[Detection]:
        """The same, on an image already in memory.

        Exists so `ape.perturb` can degrade a frame and score the result without
        writing it to disk first. `detect` is a thin wrapper over this, so the
        two cannot drift: a robustness sweep that ran a different pipeline from
        the baseline would be measuring the pipeline rather than the
        perturbation.
        """
        import numpy as np

        tensor, box = self._preprocess(image)

        raw = self.session.run(None, {self.input_name: tensor})[0]
        # YOLOv8 emits (1, 4 + classes, anchors): box first, then class scores,
        # with no separate objectness. Transposed here so one row is one anchor.
        predictions = np.squeeze(raw, 0).T

        scores = predictions[:, 4:]
        best = scores.argmax(axis=1)
        confidence = scores[np.arange(len(scores)), best]
        alive = confidence >= self.score_threshold

        rows = predictions[alive]
        classes = best[alive]
        kept_scores = confidence[alive]

        # Corner form and image coordinates for everything at once. Doing this
        # per box in Python is what made the first version quadratic in wall
        # clock as well as in comparisons.
        cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
        corners = np.stack([(cx - w / 2 - box.pad_x) / box.scale,
                            (cy - h / 2 - box.pad_y) / box.scale,
                            (cx + w / 2 - box.pad_x) / box.scale,
                            (cy + h / 2 - box.pad_y) / box.scale], axis=1)

        # NMS per class, not globally. A pedestrian standing in front of a car
        # overlaps it heavily, and suppressing across classes would delete one
        # of two correct answers.
        # Grouped by the KITTI label, NOT by the COCO class. `bicycle` and
        # `motorcycle` both become Cyclist, and `truck` and `bus` both become
        # Truck, so suppressing per COCO class would leave two boxes on the same
        # object, each surviving because the other was in a different group.
        mapped = np.array([to_kitti(COCO_NAMES[int(c)]) or "" for c in classes])

        out: list[Detection] = []
        for label in (name for name in np.unique(mapped) if name):
            mine = mapped == label
            boxes, scores = corners[mine], kept_scores[mine]
            for index in _nms(boxes, scores, self.iou_threshold):
                out.append(Detection(frame_id=frame_id, label=label,
                                     box=Box2D(*(float(v) for v in boxes[index])),
                                     score=float(scores[index])))
        out.sort(key=lambda d: d.score, reverse=True)
        return out
