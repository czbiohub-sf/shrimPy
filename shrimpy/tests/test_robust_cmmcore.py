"""Tests for RobustCMMCore.

Unit tests cover _make_robust_call retry logic in isolation.
Structural tests verify RobustCMMCore wraps correctly and passes through
non-callable / private attributes unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pymmcore_plus.core import CMMCorePlus

from shrimpy.robust_cmmcore import (
    NUM_RETRIES,
    WAIT_BETWEEN_RETRIES_S,
    RobustCMMCore,
    _make_robust_call,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def core() -> RobustCMMCore:
    """RobustCMMCore without a loaded hardware config."""
    return RobustCMMCore()


# ---------------------------------------------------------------------------
# _make_robust_call — unit tests
# ---------------------------------------------------------------------------


def test_robust_call_returns_result_on_first_try():
    func = MagicMock(return_value=42)
    wrapped = _make_robust_call("test", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
    assert wrapped() == 42
    assert func.call_count == 1


def test_robust_call_retries_on_failure_then_succeeds():
    func = MagicMock(side_effect=[RuntimeError("e1"), RuntimeError("e2"), "ok"])
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        wrapped = _make_robust_call("test", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
        result = wrapped()
    assert result == "ok"
    assert func.call_count == 3


def test_robust_call_raises_after_all_retries_exhausted():
    func = MagicMock(side_effect=RuntimeError("permanent"))
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        wrapped = _make_robust_call("test", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
        with pytest.raises(RuntimeError, match="permanent"):
            wrapped()
    assert func.call_count == NUM_RETRIES


def test_robust_call_sleeps_between_retries_not_after_last():
    func = MagicMock(side_effect=RuntimeError("e"))
    with patch("shrimpy.robust_cmmcore.time.sleep") as mock_sleep:
        wrapped = _make_robust_call("test", func, NUM_RETRIES, 5.0)
        with pytest.raises(RuntimeError):
            wrapped()
    # sleep is called between attempts only: NUM_RETRIES - 1 times
    assert mock_sleep.call_count == NUM_RETRIES - 1
    mock_sleep.assert_called_with(5.0)


def test_robust_call_no_sleep_on_first_try_success():
    func = MagicMock(return_value="ok")
    with patch("shrimpy.robust_cmmcore.time.sleep") as mock_sleep:
        wrapped = _make_robust_call("test", func, NUM_RETRIES, 5.0)
        wrapped()
    mock_sleep.assert_not_called()


def test_robust_call_reraises_last_exception():
    func = MagicMock(
        side_effect=[RuntimeError("first"), RuntimeError("second"), RuntimeError("last")]
    )
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        wrapped = _make_robust_call("test", func, NUM_RETRIES, 0.0)
        with pytest.raises(RuntimeError, match="last"):
            wrapped()


def test_robust_call_passes_args_and_kwargs():
    func = MagicMock(return_value="ok")
    wrapped = _make_robust_call("test", func, NUM_RETRIES, 0.0)
    wrapped("a", "b", key="val")
    func.assert_called_once_with("a", "b", key="val")


def test_robust_call_excluded_method_not_retried():
    # getMultiROI should be called once even if it raises
    func = MagicMock(side_effect=RuntimeError("fail"))
    wrapped = _make_robust_call("getMultiROI", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
    with pytest.raises(RuntimeError):
        wrapped()
    assert func.call_count == 1


def test_robust_call_excluded_property_args_not_retried():
    # getProperty('TS2_TTL1-8', 'Label') should be called once even if it raises
    func = MagicMock(side_effect=RuntimeError("fail"))
    wrapped = _make_robust_call("getProperty", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
    with pytest.raises(RuntimeError):
        wrapped("TS2_TTL1-8", "Label")
    assert func.call_count == 1


def test_robust_call_unhashable_args_still_retried():
    # If args contain an unhashable type the exclusion check must not crash,
    # and the call should still be retried normally.
    func = MagicMock(side_effect=RuntimeError("fail"))
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        wrapped = _make_robust_call("getProperty", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
        with pytest.raises(RuntimeError):
            wrapped({"unhashable": "dict"}, "Label")
    assert func.call_count == NUM_RETRIES


def test_robust_call_getproperty_other_args_still_retried():
    # getProperty with different args should still be retried
    func = MagicMock(side_effect=RuntimeError("fail"))
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        wrapped = _make_robust_call("getProperty", func, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
        with pytest.raises(RuntimeError):
            wrapped("SomeDevice", "SomeProperty")
    assert func.call_count == NUM_RETRIES


def test_robust_call_num_retries_one_tries_exactly_once():
    func = MagicMock(side_effect=RuntimeError("e"))
    with patch("shrimpy.robust_cmmcore.time.sleep") as mock_sleep:
        wrapped = _make_robust_call("test", func, 1, 5.0)
        with pytest.raises(RuntimeError):
            wrapped()
    assert func.call_count == 1
    mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# RobustCMMCore — structural / behavioural tests
# ---------------------------------------------------------------------------


def test_is_instance_of_cmmcoreplus(core):
    assert isinstance(core, CMMCorePlus)


def test_is_instance_of_robust_cmmcore(core):
    assert isinstance(core, RobustCMMCore)


def test_robust_ready_flag_set_after_init(core):
    assert core._robust_ready is True


def test_public_method_is_wrapped(core):
    # functools.wraps sets __wrapped__ on the closure returned by _make_robust_call
    assert hasattr(core.getLoadedDevices, "__wrapped__")


def test_private_attribute_not_wrapped(core):
    # _robust_ready is underscore-prefixed → returned as-is (not callable anyway)
    assert core._robust_ready is True


def test_events_not_wrapped(core):
    # events returns CMMCoreSignaler which is not callable → passes through
    assert not callable(core.events)
    assert not hasattr(core.events, "__wrapped__")


def test_mda_not_wrapped(core):
    # mda returns MDARunner which is not callable → passes through
    assert not callable(core.mda)
    assert not hasattr(core.mda, "__wrapped__")


def test_real_method_call_works(core):
    # Verify the wrapper doesn't break real method calls
    paths = core.getDeviceAdapterSearchPaths()
    assert isinstance(paths, (list, tuple))


def test_retries_on_failure_through_core(core):
    # Inject a flaky function and confirm it is retried via __getattribute__
    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < NUM_RETRIES:
            raise RuntimeError("transient")
        return "recovered"

    object.__setattr__(core, "getLoadedDevices", flaky)
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        result = core.getLoadedDevices()

    assert result == "recovered"
    assert call_count == NUM_RETRIES


def test_raises_after_all_retries_through_core(core):
    func = MagicMock(side_effect=RuntimeError("always fails"))
    object.__setattr__(core, "getLoadedDevices", func)
    with patch("shrimpy.robust_cmmcore.time.sleep"):
        with pytest.raises(RuntimeError, match="always fails"):
            core.getLoadedDevices()
    assert func.call_count == NUM_RETRIES
