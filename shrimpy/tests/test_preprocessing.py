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
    """A pipeline lacking both 'deskew' and 'phase' cannot reconstruct -> None.

    Also assert it warns. We attach a handler directly to the module logger
    rather than using ``caplog`` (which captures at root): shrimpy's logging
    config sets ``propagate=False`` on the ``shrimpy`` logger, so root-level
    capture is unreliable when the full suite runs.
    """
    import logging

    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    module_logger = logging.getLogger("shrimpy.preprocessing")
    module_logger.addHandler(handler)
    try:
        assert build_preprocessor(ZYX, ["vs"]) is None
        assert build_preprocessor(ZYX, ["sum_projection"]) is None
    finally:
        module_logger.removeHandler(handler)

    assert any("requires a 'deskew' and/or 'phase' step" in r.getMessage() for r in records)


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
