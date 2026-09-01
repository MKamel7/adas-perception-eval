"""Metamorphic perturbations: the same scene, degraded a stated amount.

THE QUESTION THIS ANSWERS, and why it is not "how does the detector do on
another dataset". A second dataset changes the scene, the camera, the labelling
policy and the class balance all at once, so a drop in AP has four candidate
causes and the result is a number rather than a finding. A perturbation changes
exactly one thing by a stated amount and keeps the ground truth identical, so
the curve of AP against strength is attributable.

That property, ground truth unchanged, is what makes these metamorphic
relations rather than augmentations. Every perturbation here is one a camera
actually suffers, and none of them moves an object:

  brightness   exposure error, or a tunnel mouth
  contrast     haze, low sun, a dirty windscreen
  blur         defocus, or motion at speed
  jpeg         a compressed video pipeline between sensor and detector
  fog          scattering, as a uniform veil

NOT INCLUDED, DELIBERATELY: crop. It is a reasonable perturbation and it moves
the boxes, so the ground truth would have to be transformed with it. That makes
it a different experiment, one where a bug in the box transform is
indistinguishable from a real drop, and the whole point of this module is that
nothing about the labels changes.

THE FOG IS NOT PHYSICAL, and the report says so. Real fog attenuates with
distance, so it should hurt a pedestrian at 40 m far more than one at 5 m. This
is a uniform veil, which is the depth-independent approximation, so it
understates the distance dependence that matters most for ADAS. It is evidence
about a veil, not about weather. `vkitti` is where depth-aware weather lives.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

#: Strength 0 must be the identity for every perturbation, so the baseline of a
#: sweep is the unperturbed run and any difference at 0 is a bug in the harness
#: rather than a result.
IDENTITY = 0.0


def brightness(image: Any, strength: float) -> Any:
    """Scale luminance. strength is the fractional change, +0.4 is 40% brighter."""
    from PIL import ImageEnhance
    return ImageEnhance.Brightness(image).enhance(1.0 + strength)


def contrast(image: Any, strength: float) -> Any:
    """Reduce contrast toward flat grey. strength 1.0 would be featureless."""
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(image).enhance(max(0.0, 1.0 - strength))


def blur(image: Any, strength: float) -> Any:
    """Gaussian blur; strength is the radius in pixels."""
    from PIL import ImageFilter
    if strength <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=strength))


def jpeg(image: Any, strength: float) -> Any:
    """Round-trip through JPEG. strength 0 is quality 100, 1.0 is quality 5."""
    from PIL import Image
    if strength <= 0:
        return image
    quality = max(5, round(100 - strength * 95))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("RGB")


def fog(image: Any, strength: float) -> Any:
    """Blend toward a bright grey veil.

    Uniform, not depth-aware: see the module docstring.
    """
    from PIL import Image
    if strength <= 0:
        return image
    veil = Image.new("RGB", image.size, (200, 200, 200))
    return Image.blend(image, veil, min(1.0, strength))


@dataclass(frozen=True)
class Perturbation:
    name: str
    apply: Callable[[Any, float], Any]
    #: What the strength number means, for the report axis label.
    unit: str
    #: The sweep, always starting at the identity.
    strengths: tuple[float, ...]


PERTURBATIONS: tuple[Perturbation, ...] = (
    Perturbation("brightness", brightness, "fractional change",
                 (IDENTITY, 0.3, 0.6, -0.3, -0.6)),
    Perturbation("contrast", contrast, "fraction removed",
                 (IDENTITY, 0.3, 0.6, 0.8)),
    Perturbation("blur", blur, "gaussian radius px",
                 (IDENTITY, 1.0, 2.0, 4.0)),
    Perturbation("jpeg", jpeg, "compression, 0 = quality 100",
                 (IDENTITY, 0.6, 0.85, 0.95)),
    Perturbation("fog", fog, "veil opacity",
                 (IDENTITY, 0.2, 0.4, 0.6)),
)

BY_NAME = {p.name: p for p in PERTURBATIONS}


def apply(name: str, image: Any, strength: float) -> Any:
    """Apply one named perturbation, or raise for an unknown one.

    Raising rather than returning the image unchanged: a typo in a sweep
    configuration would otherwise produce a full set of results labelled with a
    perturbation that never happened.
    """
    if name not in BY_NAME:
        raise KeyError(
            f"unknown perturbation {name!r}; known: {sorted(BY_NAME)}")
    return BY_NAME[name].apply(image, strength)
