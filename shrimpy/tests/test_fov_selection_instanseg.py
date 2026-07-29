"""Tests for the InstanSeg segmentation backend.

The backend talks to a TorchScript module, so the parts worth pinning without a real
checkpoint on disk are the ones that are pure logic and easy to get subtly wrong:
config validation, rdf.yaml pixel-size parsing, the percentile scaling / rescale /
restore contract, and the forward-kwarg type coercion TorchScript is strict about.

A fake module stands in for the real network: it records what it was called with and
returns a label map of the right shape, which is enough to assert the surrounding
contract. Tests needing the genuine 150 MB checkpoint are skipped unless
SHRIMPY_INSTANSEG_CKPT points at one.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pytest

from shrimpy.fov_selection import pipeline

torch = pytest.importorskip("torch")

REAL_CKPT = os.environ.get("SHRIMPY_INSTANSEG_CKPT")


class _FakeModule:
    """Stand-in for the InstanSeg TorchScript module.

    Returns two labelled quadrants so the caller can tell the label map apart from the
    input, and records the call so the tests can assert on shape / kwargs / selector.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, x, target_segmentation=None, **kwargs):
        self.calls.append(
            {
                "shape": tuple(x.shape),
                "dtype": x.dtype,
                "target_segmentation": target_segmentation,
                "kwargs": kwargs,
            }
        )
        h, w = x.shape[-2:]
        out = torch.zeros((1, 1, h, w), dtype=torch.float32)
        out[..., : h // 2, : w // 2] = 1
        out[..., h // 2 :, w // 2 :] = 2
        return out

    def eval(self):
        return self


def _fake_model(pixel_size_um: float | None = 0.35) -> pipeline._InstansegModel:
    return pipeline._InstansegModel(_FakeModule(), pixel_size_um, "cpu")


# --- rdf.yaml pixel size ------------------------------------------------------------


def _rdf(scale, unit="micrometer"):
    return {
        "inputs": [{"axes": [{"type": "batch"}, {"id": "y", "scale": scale, "unit": unit}]}]
    }


def test_rdf_pixel_size_read_from_spatial_axis():
    assert pipeline._instanseg_rdf_pixel_size(_rdf(0.35)) == 0.35


def test_rdf_pixel_size_ignores_non_micrometer_units():
    # A scale in another unit must NOT be read as microns -- that would silently rescale
    # every FOV by a wrong factor, which is far worse than not rescaling at all.
    assert pipeline._instanseg_rdf_pixel_size(_rdf(0.35, unit="nanometer")) is None


def test_rdf_pixel_size_missing_or_malformed_returns_none():
    assert pipeline._instanseg_rdf_pixel_size({}) is None
    assert pipeline._instanseg_rdf_pixel_size({"inputs": []}) is None
    assert pipeline._instanseg_rdf_pixel_size(_rdf(None)) is None


# --- segmentation contract ----------------------------------------------------------


def test_mask_returned_on_the_original_pixel_grid():
    # Downstream features (coverage_frac, max_gap_um) are computed against the acquisition
    # px_um, so the mask must come back at the input's shape no matter how it was rescaled.
    model = _fake_model(0.35)
    img = np.random.default_rng(0).normal(300, 40, (512, 400)).astype(np.float32)
    mask = pipeline._segment_instanseg(img, model, {"model": "instanseg"}, px_um=0.1133)

    assert mask.shape == img.shape
    assert mask.dtype == np.uint32
    # ...while the network itself saw the rescaled size.
    assert model.module.calls[0]["shape"] == (1, 1, 166, 129)


def test_input_is_float32_not_promoted_by_percentile():
    # np.percentile returns float64; letting it promote the image reaches TorchScript as a
    # DoubleTensor and fails against float weights.
    model = _fake_model(0.35)
    img = np.ones((64, 64), np.float32)
    img[:32] = 5.0
    pipeline._segment_instanseg(img, model, {"model": "instanseg"}, px_um=0.35)
    assert model.module.calls[0]["dtype"] == torch.float32


def test_no_rescale_when_pixel_sizes_match_or_are_unknown():
    img = np.random.default_rng(1).normal(0, 1, (64, 80)).astype(np.float32)

    same = _fake_model(0.35)
    pipeline._segment_instanseg(img, same, {"model": "instanseg"}, px_um=0.35)
    assert same.module.calls[0]["shape"] == (1, 1, 64, 80)

    unknown = _fake_model(None)  # bare .pt with no model_pixel_size_um
    pipeline._segment_instanseg(img, unknown, {"model": "instanseg"}, px_um=0.1133)
    assert unknown.module.calls[0]["shape"] == (1, 1, 64, 80)

    no_px = _fake_model(0.35)  # px_um not supplied by the caller
    pipeline._segment_instanseg(img, no_px, {"model": "instanseg"}, px_um=None)
    assert no_px.module.calls[0]["shape"] == (1, 1, 64, 80)


def test_rescale_never_shrinks_below_model_minimum():
    # A small FOV at a much finer pixel size would otherwise scale to under the network's
    # 32 px minimum input and fail inside TorchScript.
    model = _fake_model(0.35)
    img = np.zeros((40, 40), np.float32)
    pipeline._segment_instanseg(img, model, {"model": "instanseg"}, px_um=0.01)
    assert model.module.calls[0]["shape"][-2:] == (
        pipeline.INSTANSEG_MIN_SIZE_PX,
        pipeline.INSTANSEG_MIN_SIZE_PX,
    )


def test_target_selects_the_output_head():
    img = np.zeros((64, 64), np.float32)
    for target, expected in (("nuclei", [1, 0]), ("cells", [0, 1])):
        model = _fake_model(0.35)
        pipeline._segment_instanseg(
            img, model, {"model": "instanseg", "target": target}, px_um=0.35
        )
        sel = model.module.calls[0]["target_segmentation"]
        assert sel.tolist() == expected


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="segmentation.target"):
        pipeline._segment_instanseg(
            np.zeros((64, 64), np.float32),
            _fake_model(),
            {"model": "instanseg", "target": "mitochondria"},
            px_um=0.35,
        )


def test_forward_kwargs_are_coerced_to_the_types_torchscript_demands():
    # YAML happily gives an int where the schema says float (and vice versa); TorchScript
    # rejects that outright, so the backend must coerce rather than pass values through.
    model = _fake_model(0.35)
    seg = {
        "model": "instanseg",
        "min_size": 20.0,  # float in YAML -> must reach the module as int
        "mask_threshold": 1,  # int in YAML -> must reach the module as float
        "tta": False,
    }
    pipeline._segment_instanseg(np.zeros((64, 64), np.float32), model, seg, px_um=0.35)

    kwargs = model.module.calls[0]["kwargs"]
    assert isinstance(kwargs["min_size"], int) and kwargs["min_size"] == 20
    assert isinstance(kwargs["mask_threshold"], float) and kwargs["mask_threshold"] == 1.0
    assert kwargs["tta"] is False
    # Unset knobs must not be forwarded at all, so the model keeps its own defaults.
    assert set(kwargs) == {"min_size", "mask_threshold", "tta"}


def test_segment_2d_dispatches_to_instanseg():
    model = _fake_model(0.35)
    mask = pipeline.segment_2d(
        np.zeros((64, 64), np.float32),
        model,
        "brightfield",
        {"model": "instanseg"},
        px_um=0.35,
    )
    assert model.module.calls and mask.shape == (64, 64)


# --- loader validation --------------------------------------------------------------


def test_load_segmenter_requires_a_path():
    with pytest.raises(ValueError, match="requires a 'path'"):
        pipeline.load_segmenter({"model": "instanseg"})


def test_load_segmenter_rejects_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.load_segmenter({"model": "instanseg", "path": str(tmp_path / "nope.zip")})


def test_load_segmenter_rejects_unknown_backend():
    with pytest.raises(NotImplementedError, match="instanseg"):
        pipeline.load_segmenter({"model": "stardist"})


# --- against the real checkpoint (opt-in) -------------------------------------------


@pytest.mark.skipif(not REAL_CKPT, reason="set SHRIMPY_INSTANSEG_CKPT to a checkpoint")
def test_real_checkpoint_round_trip():
    seg = {"model": "instanseg", "path": REAL_CKPT, "gpu": False}
    model = pipeline.load_segmenter(seg)
    assert model.pixel_size_um == pytest.approx(0.35)

    img = np.random.default_rng(0).normal(300, 40, (256, 256)).astype(np.float32)
    mask = pipeline.segment_2d(img, model, "brightfield", seg, px_um=0.35)
    assert mask.shape == img.shape and mask.dtype == np.uint32


@pytest.mark.skipif(not REAL_CKPT, reason="set SHRIMPY_INSTANSEG_CKPT to a checkpoint")
def test_real_checkpoint_is_scale_invariant():
    # The same scene presented at 2x resolution and half the pixel size must segment the
    # same -- this is the property the rescale step exists to provide.
    seg = {"model": "instanseg", "path": REAL_CKPT, "gpu": False}
    model = pipeline.load_segmenter(seg)

    import io
    import zipfile

    with zipfile.ZipFile(Path(REAL_CKPT)) as zf:
        img = np.load(io.BytesIO(zf.read("test-input.npy")))[0, 0]

    native = pipeline.segment_2d(img, model, "bf", seg, px_um=0.35)
    upscaled = pipeline.segment_2d(
        np.repeat(np.repeat(img, 2, 0), 2, 1), model, "bf", seg, px_um=0.175
    )
    assert int(np.unique(native).size) == int(np.unique(upscaled).size)
    assert (native > 0).mean() == pytest.approx((upscaled > 0).mean(), abs=0.01)
