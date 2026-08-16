"""Tests for the shared label-free preprocessing pipeline.

Focus on the wiring FOV selection (and DynaTrack) rely on:

* ``build_preprocessor`` step selection (which pipelines produce a preprocessor
  and which return ``None``);
* the channel-dict assembly in ``_LabelfreePreprocessor.__call__`` -- a VS
  pipeline must emit one entry per virtual-staining target channel (plus a
  ``'phase'`` debug intermediate), while a non-VS pipeline emits a single entry
  keyed by ``output_channel``.

The heavy steps (deskew / phase / VS) require GPU + model checkpoints and are
monkeypatched out; these tests exercise the pure control flow.
"""

from __future__ import annotations

import pytest

from shrimpy.preprocessing import _LabelfreePreprocessor, _settings_kwargs, build_preprocessor

ZYX = (16, 64, 64)


def test_no_pipeline_returns_none():
    """No preprocessing steps -> no preprocessor."""
    assert build_preprocessor(ZYX, None) is None
    assert build_preprocessor(ZYX, []) is None


def test_pipeline_without_reconstruction_step_returns_none():
    """A pipeline with no reconstruction step (only projection/segmentation) -> None.

    No step is mandatory: the caller uses the raw stack directly when nothing needs
    reconstructing.
    """
    assert build_preprocessor(ZYX, ["sum_projection"]) is None
    assert build_preprocessor(ZYX, ["middle_slice_projection", "segmentation"]) is None


def test_any_reconstruction_step_builds_a_preprocessor():
    """flatfield / deskew / phase / vs each build a preprocessor, in any combination.

    Presence of a step is data-driven -- deskew and phase are not individually required.
    """
    pytest.importorskip("torch")

    for pipeline in (["flatfield"], ["deskew"], ["vs"], ["flatfield", "best_focus_z"]):
        assert build_preprocessor(ZYX, pipeline) is not None, pipeline


def test_settings_kwargs_filters_to_signature():
    """_settings_kwargs keeps only fields the target callable accepts."""

    class FakeSettings:
        def model_dump(self):
            return {"a": 1, "b": 2, "unused": 3}

    def func(a, b):  # noqa: D401 - signature only
        return a, b

    assert _settings_kwargs(func, FakeSettings()) == {"a": 1, "b": 2}


def _bare_preprocessor(**kwargs):
    """Construct a _LabelfreePreprocessor without running warm_up()."""
    defaults = dict(
        zyx_shape=ZYX,
        deskew_settings=None,
        phase_settings=None,
        vs_config=None,
        output_channel="phase",
    )
    defaults.update(kwargs)
    pre = _LabelfreePreprocessor(**defaults)
    pre._device = None  # keep tensors on CPU
    pre._log_gpu_memory = staticmethod(lambda: None)  # avoid torch cuda probing
    return pre


def test_call_non_vs_keys_output_channel():
    """A non-VS pipeline returns a single entry keyed by output_channel."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    pre = _bare_preprocessor(output_channel="BF")
    out = pre(np.zeros(ZYX, dtype="float32"))

    assert set(out) == {"BF"}
    assert isinstance(out["BF"], torch.Tensor)
    assert tuple(out["BF"].shape) == ZYX


def test_call_vs_pipeline_assembles_target_channels(monkeypatch):
    """A VS pipeline emits one entry per target channel plus a 'phase' debug key.

    This is the exact contract FOV selection depends on: it segments the
    ``nuclei``/``membrane`` channels returned here.
    """
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    # phase_settings/vs_config just need to be truthy to select the branches;
    # the heavy steps are monkeypatched.
    pre = _bare_preprocessor(phase_settings=object(), vs_config={"dummy": True})
    monkeypatch.setattr(pre, "_reconstruct_phase", lambda vol: vol)
    monkeypatch.setattr(
        pre,
        "_predict_vs",
        lambda vol: {"nuclei": torch.zeros(ZYX), "membrane": torch.zeros(ZYX)},
    )

    out = pre(np.zeros(ZYX, dtype="float32"))

    assert set(out) == {"phase", "nuclei", "membrane"}


def test_require_gpu_raises_when_device_is_cpu(monkeypatch):
    """FOV selection (require_gpu=True) must fail fast on a CPU-only machine."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("shrimpy.preprocessing._resolve_device", lambda: torch.device("cpu"))

    with pytest.raises(RuntimeError, match="GPU required but none detected"):
        build_preprocessor(ZYX, ["phase"], require_gpu=True)


def test_require_gpu_false_allows_cpu(monkeypatch):
    """DynaTrack / default callers keep working on CPU (no fail-fast)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("shrimpy.preprocessing._resolve_device", lambda: torch.device("cpu"))

    # Default require_gpu=False: builds without raising on CPU.
    assert build_preprocessor(ZYX, ["phase"], require_gpu=False) is not None


def test_flat_field_bf_matches_biahub_reference():
    """The torch (GPU-capable) flat-field matches biahub's numpy reference.

    Uses an even Z count, where numpy median averages the two middle values --
    the case torch.median gets wrong but Tensor.quantile(0.5) gets right.
    """
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    ff = pytest.importorskip("biahub.flat_field_correction")

    rng = np.random.default_rng(3)
    vol = rng.integers(80, 600, (8, 6, 10)).astype(np.float32)  # even Z
    ref = ff.flat_field_correction(vol, axis=0).astype(np.float32)

    pre = _bare_preprocessor()
    out = pre._flat_field_BF(torch.as_tensor(vol, dtype=torch.float32)).numpy()

    assert np.allclose(ref, out, atol=1e-2)
