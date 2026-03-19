"""CMMCorePlus subclass that adds retry logic to hardware calls."""

from __future__ import annotations

import functools
import logging
import time

from pymmcore_plus.core import CMMCorePlus

logger = logging.getLogger(__name__)

NUM_RETRIES = 3
WAIT_BETWEEN_RETRIES_S = 5.0  # seconds


_NO_RETRY_CALLS: set[str] = {"getMultiROI", "fullFocus"}
_NO_RETRY_CALLS_WITH_ARGS: set[tuple] = {
    ("getProperty", "TS2_TTL1-8", "Label"),
    ("getStateLabels", "TS2_TTL1-8"),
}


def _make_robust_call(name: str, func, num_retries: int, wait_s: float):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if name in _NO_RETRY_CALLS:
            return func(*args, **kwargs)
        try:
            if (name, *args) in _NO_RETRY_CALLS_WITH_ARGS:
                return func(*args, **kwargs)
        except TypeError:
            pass  # args contain an unhashable type; proceed with retry logic

        last_error: Exception | None = None
        for attempt in range(num_retries):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.debug(f"Call to {name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_error = e
                msg = str(e).split("\n")[0]
                logger.debug(
                    f"Attempt {attempt + 1}/{num_retries} failed calling {name} with arguments {args} and keywords {kwargs}: {msg}"
                )
                if attempt < num_retries - 1:
                    time.sleep(wait_s)
        logger.critical(f"Call to {name} failed after {num_retries} attempts")
        raise last_error  # type: ignore[misc]

    return wrapper


class RobustCMMCore(CMMCorePlus):
    """CMMCorePlus subclass that retries failed method calls.

    Every public method call (non-underscore-prefixed callable) is attempted
    up to NUM_RETRIES times with WAIT_BETWEEN_RETRIES_S seconds between
    attempts.  Non-callable attributes and private/dunder attributes are
    forwarded without modification.

    Usage is identical to CMMCorePlus:

        core = RobustCMMCore()
        core.loadSystemConfiguration("path/to/config.cfg")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set only after parent init completes so that __getattribute__ does not
        # intercept internal bound-method lookups (e.g. weakref.WeakMethod) during init.
        object.__setattr__(self, "_robust_ready", True)

    def __getattribute__(self, name: str):
        attr = super().__getattribute__(name)
        try:
            ready = object.__getattribute__(self, "_robust_ready")
        except AttributeError:
            ready = False
        if ready and callable(attr) and not name.startswith("_"):
            return _make_robust_call(name, attr, NUM_RETRIES, WAIT_BETWEEN_RETRIES_S)
        return attr
