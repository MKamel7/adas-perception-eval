"""The perturbations do one stated thing, and strength 0 does nothing.

Both properties are what make a degradation curve attributable. If strength 0
were not the identity, the baseline of every sweep would be wrong; if a
perturbation changed the image size or moved content, the ground truth would no
longer describe the frame and the curve would be measuring a labelling error.
"""

from __future__ import annotations

import pytest

from ape.perturb import BY_NAME, PERTURBATIONS, apply

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


def scene(width: int = 96, height: int = 64):
    """A deterministic image with real structure, not flat colour.

    A flat image is invariant under blur and contrast, so it would let a
    broken perturbation pass every test here.
    """
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    array[16:48, 24:72] = 240   # a bright block, so contrast has something to flatten
    return Image.fromarray(array, "RGB")


def as_array(image):
    return np.asarray(image, dtype=np.float64)


@pytest.mark.parametrize("perturbation", PERTURBATIONS, ids=lambda p: p.name)
def test_strength_zero_is_the_identity(perturbation):
    """The baseline of every sweep depends on this being exactly true."""
    original = scene()

    result = perturbation.apply(original, 0.0)

    assert np.array_equal(as_array(result), as_array(original))


@pytest.mark.parametrize("perturbation", PERTURBATIONS, ids=lambda p: p.name)
def test_the_first_strength_in_every_sweep_is_the_identity(perturbation):
    assert perturbation.strengths[0] == 0.0


@pytest.mark.parametrize("perturbation", PERTURBATIONS, ids=lambda p: p.name)
def test_size_and_mode_are_preserved(perturbation):
    """A perturbation that resized the frame would invalidate every box."""
    original = scene()

    for strength in perturbation.strengths:
        result = perturbation.apply(original, strength)
        assert result.size == original.size, perturbation.name
        assert result.mode == "RGB", perturbation.name


@pytest.mark.parametrize("perturbation", PERTURBATIONS, ids=lambda p: p.name)
def test_a_nonzero_strength_actually_changes_the_image(perturbation):
    """Otherwise the curve would be flat for a reason that is not the detector."""
    original = scene()
    strongest = perturbation.strengths[-1]

    result = perturbation.apply(original, strongest)

    assert not np.array_equal(as_array(result), as_array(original))


@pytest.mark.parametrize("perturbation", PERTURBATIONS, ids=lambda p: p.name)
def test_perturbations_are_deterministic(perturbation):
    """A sweep is compared against a baseline run separately; both must repeat."""
    original = scene()
    strongest = perturbation.strengths[-1]

    first = as_array(perturbation.apply(original, strongest))
    second = as_array(perturbation.apply(original, strongest))

    assert np.array_equal(first, second)


# ---- each one does the specific thing it claims -----------------------------

def test_brightness_moves_the_mean_in_the_signed_direction():
    original = scene()
    base = as_array(original).mean()

    assert as_array(apply("brightness", original, 0.6)).mean() > base
    assert as_array(apply("brightness", original, -0.6)).mean() < base


def test_contrast_reduces_the_spread():
    original = scene()

    assert as_array(apply("contrast", original, 0.8)).std() < as_array(original).std()


def test_blur_reduces_local_gradient_and_more_so_with_radius():
    """Monotonic in strength, which is what makes the x axis mean anything."""
    original = scene()

    def gradient(image):
        a = as_array(image).mean(axis=2)
        return float(np.abs(np.diff(a, axis=1)).mean())

    sharp = gradient(original)
    mild = gradient(apply("blur", original, 1.0))
    heavy = gradient(apply("blur", original, 4.0))

    assert heavy < mild < sharp


def test_jpeg_gets_further_from_the_original_as_it_compresses_harder():
    original = scene()

    def distance(strength):
        return float(np.abs(as_array(apply("jpeg", original, strength))
                            - as_array(original)).mean())

    assert distance(0.95) > distance(0.6) > 0


def test_fog_moves_everything_toward_the_veil():
    """A veil raises the darks and lowers the brights: the spread collapses."""
    original = scene()

    veiled = as_array(apply("fog", original, 0.6))

    assert veiled.std() < as_array(original).std()
    assert veiled.min() > as_array(original).min()


# ---- the registry -----------------------------------------------------------

def test_an_unknown_perturbation_raises_rather_than_passing_the_image_through():
    """A typo in a sweep config would otherwise produce a full set of results
    labelled with a perturbation that never happened."""
    with pytest.raises(KeyError, match="unknown perturbation"):
        apply("motion_blurr", scene(), 1.0)


def test_every_registered_perturbation_is_reachable_by_name():
    assert set(BY_NAME) == {p.name for p in PERTURBATIONS}
