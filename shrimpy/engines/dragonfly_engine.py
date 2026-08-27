"""Acquisition engine for the Dragonfly microscope.

Inherits the shared behavior from
:class:`~shrimpy.engines.base_engine.BaseEngine` (hardware sequencing defaults,
hardware logging, autofocus dispatch and event skipping, hardware reset on
teardown, and ``acquire()``) and adds the Leica Adaptive Focus Control (AFC)
autofocus routine. See :mod:`shrimpy.engines.mantis_engine` for a more
elaborate subclass.

Unlike mantis, Dragonfly acquisitions are usually not hardware-sequenced, so
``setup_event`` mostly receives single events, one per Z slice; AFC therefore
engages on arrival at each position rather than once per burst, and the lock
is reused for every channel and slice acquired there. Both paths are handled by
:meth:`BaseEngine._should_engage_autofocus`.
"""

from __future__ import annotations

import logging
import time

from typing import TYPE_CHECKING

from pymmcore_plus import CMMCorePlus, DeviceType

from shrimpy.engines.base_engine import BaseEngine

if TYPE_CHECKING:
    from useq import MDAEvent

logger = logging.getLogger(__name__)

# Z-related devices worth snapshotting alongside the configured autofocus stage.
# On the Dragonfly, ``FocusDrive`` (Leica coarse drive) and the AFC share COM4
# and the same LeicaDMI hub, while ``ZStage`` (ASI piezo) is the Core-Focus
# device that carries the Z-stack. A native Micro-Manager run that focuses
# reliably never moves either one before ``fullFocus()``, so recording all of
# them makes it obvious which stage shrimPy actually disturbed.
AFC_RELATED_DEVICES = (
    "FocusDrive",
    "ZStage",
    "Adaptive Focus Control",
    "Adaptive Focus Control Offset",
)

# Reference durations measured in a native Micro-Manager acquisition on this
# scope where AFC locked on the first attempt every time (20/20). Logged next to
# the measured values so an anomaly is obvious without cross-referencing.
NATIVE_FULL_FOCUS_S = 0.24
NATIVE_WAIT_FOR_Z_STAGE_S = 0.05

# Smallest focus-stage move worth commanding, in um. Below this the stage is
# treated as already at the target and no command is sent at all; see
# _move_focus_stage() for why a zero-distance move is not merely wasteful but
# hangs. 0.1 um is far below the depth of field of any objective on this scope,
# so skipping such a move cannot change where the sample is imaged.
MIN_FOCUS_MOVE_UM = 0.1


class DragonflyEngine(BaseEngine):
    """MDA engine for the Dragonfly microscope.

    Hooks left to implement:

    - ``setup_sequence`` / ``setup_event`` / ``teardown_sequence``: Dragonfly
      hardware setup around the corresponding ``super()`` call
    """

    def engage_autofocus(self, event: MDAEvent) -> bool:
        """Engage Leica AFC for ``event``.

        Called by :meth:`~shrimpy.engines.base_engine.BaseEngine._engage_autofocus`
        once per position (or once per burst, if the event was sequenced) when
        autofocus is enabled with a method other than ``demo-PFS``. The
        acquisition of events for which this returns False is skipped.
        """
        z_position = self._get_autofocus_z_position(event)
        logger.debug(
            f"engage_autofocus: event index={dict(event.index)}, "
            f"event.z_pos={event.z_pos}, resolved autofocus z={z_position}"
        )
        return self._engage_leica_afc(self._autofocus_stage, z_position)

    def _raw_core_call(self, method: str, *args):
        """Call ``method`` on the core, bypassing any retry wrapper.

        Diagnostic queries must fail fast. Under
        :class:`~shrimpy.robust_cmmcore.RobustCMMCore` every failing call is
        retried 3 times with 5 s between attempts, so a query that can never
        succeed — asking an AutoFocus device for its position, say — costs 10 s
        and a CRITICAL log line instead of microseconds. Resolving the method
        against :class:`CMMCorePlus` skips that layer.
        """
        core = self.mmcore
        func = getattr(CMMCorePlus, method, None)
        if func is not None:
            try:
                return func(core, *args)
            except TypeError:
                # Not a real CMMCore instance (a mock, or some other wrapper):
                # the unbound descriptor refuses it. Fall back to the bound
                # attribute, retry wrapper and all.
                pass
        return getattr(core, method)(*args)

    def _log_afc_state(self, context: str) -> None:
        """Log everything we can cheaply query about the AFC / Z stage state.

        Purely diagnostic. Every query is guarded individually so that one
        device that raises does not hide the state we did manage to read.
        """
        stage = self._autofocus_stage
        af_device = self._autofocus_method

        def _safe(label: str, method: str, *args) -> None:
            try:
                logger.debug(f"  [{context}] {label}: {self._raw_core_call(method, *args)!r}")
            except Exception as exc:
                logger.debug(
                    f"  [{context}] {label}: <query failed: {type(exc).__name__}: {exc}>"
                )

        _safe("focus device (Core-Focus)", "getFocusDevice")
        _safe("autofocus device", "getAutoFocusDevice")
        _safe("system busy", "systemBusy")
        _safe("continuous focus enabled", "isContinuousFocusEnabled")
        _safe("continuous focus locked", "isContinuousFocusLocked")
        _safe("last focus score", "getLastFocusScore")
        _safe("current focus score", "getCurrentFocusScore")

        # Position and busy state of every Z-related device, not just the
        # configured autofocus stage: AFC drives the Leica FocusDrive itself,
        # and the Z-stack rides the ASI piezo, so the interesting question is
        # which of them is moving (or stuck) when the wait fails.
        #
        # Only Stage devices get getPosition(): the AFC itself is an AutoFocus
        # device and raises "wrong type for the requested operation", which the
        # RobustCMMCore retry wrapper would turn into a 10 s stall per query.
        try:
            loaded = set(self._raw_core_call("getLoadedDevices"))
        except Exception:
            logger.exception(f"  [{context}] could not list loaded devices")
            loaded = set()
        for device in dict.fromkeys((stage, *AFC_RELATED_DEVICES)):
            if not device or device not in loaded:
                continue
            try:
                is_stage = self._raw_core_call("getDeviceType", device) == DeviceType.Stage
            except Exception:
                is_stage = False
            if is_stage:
                _safe(f"{device} position", "getPosition", device)
            _safe(f"{device} busy", "deviceBusy", device)

        # Full property dump of the AFC device and of the Z stage. This is
        # where Leica reports its dichroic / LED / limit / "AFC out of range"
        # state, which is the usual reason a stage never reports un-busy.
        for device in (af_device, stage):
            if not device:
                continue
            try:
                props = self._raw_core_call("getDevicePropertyNames", device)
            except Exception as exc:
                logger.debug(
                    f"  [{context}] cannot list properties of {device!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            for prop in props:
                _safe(f"{device}.{prop}", "getProperty", device, prop)

    def _poll_until_not_busy(self, device: str) -> None:
        """Log ``device``'s busy state until it clears or the core timeout passes.

        Diagnostic only, and deliberately non-fatal: the subsequent
        ``waitForDevice`` is what decides success or failure. This exists so the
        log distinguishes "the stage never reported un-busy" from
        "``deviceBusy`` itself raised" from "the stage was ready all along" —
        a bare ``waitForDevice`` timeout cannot tell those apart.
        """
        core = self.mmcore
        try:
            timeout_s = float(core.getTimeoutMs()) / 1000
        except Exception:
            logger.exception("  getTimeoutMs() unusable; polling for 5 s instead")
            timeout_s = 5.0
        started = time.monotonic()
        polls = 0
        while True:
            elapsed = time.monotonic() - started
            try:
                busy = core.deviceBusy(device)
            except Exception:
                logger.exception(
                    f"  deviceBusy({device}) raised after {elapsed:.3f} s and {polls} polls"
                )
                return
            polls += 1
            if not busy:
                logger.debug(
                    f"  {device} reported not busy after {elapsed:.3f} s ({polls} poll(s))"
                )
                return
            if elapsed > timeout_s:
                logger.warning(
                    f"  {device} still busy after {elapsed:.3f} s and {polls} polls "
                    f"(core timeout {timeout_s:.3f} s); waitForDevice is expected "
                    f"to fail next"
                )
                return
            time.sleep(0.02)

    def _move_focus_stage(self, stage: str, target: float, context: str) -> bool:
        """Move ``stage`` to ``target``, skipping moves below the step threshold.

        Returns True when the stage is at ``target`` — including when it was
        already there and no command was sent — and False when the move failed,
        in which case the caller should try the next Z offset.

        Moves shorter than ``MIN_FOCUS_MOVE_UM`` are skipped rather than
        commanded, because commanding a Leica FocusDrive to the position it
        already holds *hangs*. The LeicaDMI adapter derives ``Busy()`` from the
        asynchronous ``$71004`` z-drive motion flag: it raises the flag when it
        issues the move and clears it when the motor reports stopped. A move the
        firmware resolves to zero encoder steps never moves the motor, so
        ``$71004`` never arrives, ``Busy()`` stays True, and ``waitForDevice``
        throws after the 5 s core timeout.

        The drive works in integer encoder steps, so whether a nominally-zero
        move crosses a step boundary comes down to a few nm of encoder dither.
        That made it fail on roughly 2% of engagements (1 of 45 in the
        2026-08-26 17:38 run) while the other 98% cleared in 0.000 s — which is
        exactly the intermittent ``waitForDevice`` timeout this guard removes.
        """
        core = self.mmcore

        # The distance also matters at the other extreme: a Leica coarse drive
        # travelling several mm can outlast the 5 s core timeout on its own,
        # which throws from waitForDevice and looks identical to a stuck device.
        try:
            position_before = core.getPosition(stage)
        except Exception:
            logger.exception(f"  could not read {stage} position before move ({context})")
            position_before = None

        if position_before is None:
            logger.debug(f"  {stage} distance to target unknown; commanding the move anyway")
        else:
            # Rounded before comparing so that a distance of exactly the
            # threshold still moves: abs(100.1 - 100.0) is 0.09999999999999432
            # in binary floating point. 6 decimals of a um is far finer than any
            # stage resolution, so this only cancels representation error.
            distance = round(abs(target - position_before), 6)
            logger.debug(
                f"  {stage} position before move: {position_before}, "
                f"distance to target: {distance:.4f} um"
            )
            if distance < MIN_FOCUS_MOVE_UM:
                logger.debug(
                    f"  skipping move: {distance:.4f} um is below the "
                    f"{MIN_FOCUS_MOVE_UM} um threshold, so {stage} is already on target"
                )
                return True

        move_started = time.monotonic()
        try:
            core.setPosition(stage, target)
        except Exception:
            logger.exception(
                f"  setPosition({stage}, {target}) raised after "
                f"{time.monotonic() - move_started:.3f} s ({context})"
            )
            self._log_afc_state(f"setPosition-error-{context}")
            return False
        logger.debug(f"  setPosition returned in {time.monotonic() - move_started:.3f} s")

        # Poll deviceBusy ourselves first so the log records *when* the stage
        # went un-busy (or that it never did) before waitForDevice throws. A
        # bare waitForDevice only tells us that it gave up.
        self._poll_until_not_busy(stage)

        wait_started = time.monotonic()
        try:
            core.waitForDevice(stage)
        except Exception:
            logger.exception(
                f"  waitForDevice({stage}) raised after "
                f"{time.monotonic() - wait_started:.3f} s (core timeout is "
                f"{core.getTimeoutMs()} ms) while moving to {target} ({context})"
            )
            self._log_afc_state(f"waitForDevice-error-{context}")
            return False
        logger.debug(
            f"  waitForDevice({stage}) returned in "
            f"{time.monotonic() - wait_started:.3f} s "
            f"(native MM reference ~{NATIVE_WAIT_FOR_Z_STAGE_S:.2f} s)"
        )
        try:
            logger.debug(
                f"  {stage} position after move: {core.getPosition(stage)} "
                f"(requested {target})"
            )
        except Exception:
            logger.exception(f"  could not read {stage} position after move")
        return True

    def _engage_leica_afc(self, z_stage_name: str, z_position: float) -> bool:
        """Move the Z stage to ``z_position`` and run a full AFC focus.

        Parameters
        ----------
        z_stage_name : str
            The name of the z stage device which is moved before focusing.
        z_position : float
            The target position at which autofocus will be engaged.

        Returns
        -------
        bool
            True if the AFC call succeeded.
        """
        core = self.mmcore
        z_offsets = [0, -10, 10, -20, 20, -30, 30]  # in um

        logger.debug(
            f"Engaging Leica AFC: stage={z_stage_name!r}, target z={z_position}, "
            f"offsets={z_offsets} um, core device timeout={core.getTimeoutMs()} ms"
        )

        # Which physical device are we about to move, and is it the one the
        # Z-stack rides on? A native MM run focuses with Core-Focus=ZStage (ASI
        # piezo) while the autofocus stage configured here is the Leica
        # FocusDrive, which shares COM4 with the AFC itself.
        try:
            logger.debug(
                f"  {z_stage_name}: library={core.getDeviceLibrary(z_stage_name)}, "
                f"name={core.getDeviceName(z_stage_name)}, "
                f"type={core.getDeviceType(z_stage_name)}"
            )
            focus_device = core.getFocusDevice()
            if focus_device != z_stage_name:
                logger.debug(
                    f"  note: autofocus stage {z_stage_name!r} is not the Core-Focus "
                    f"device ({focus_device!r}); the Z-stack rides {focus_device!r}"
                )
        except Exception:
            logger.exception(f"  could not identify device {z_stage_name!r}")

        self._log_afc_state("afc-entry")

        # Check if autofocus is already engaged
        try:
            already_locked = core.isContinuousFocusLocked()
        except Exception:
            logger.exception("isContinuousFocusLocked() raised; assuming not locked")
            already_locked = False
        if already_locked:
            logger.debug("Continuous autofocus is already engaged")
            return True

        afc_started = time.monotonic()
        for attempt, z_offset in enumerate(z_offsets, start=1):
            target = z_position + z_offset
            logger.debug(
                f"AFC attempt {attempt}/{len(z_offsets)}: setting position of stage "
                f"{z_stage_name} to {target} (offset {z_offset} um)"
            )

            # --- move -------------------------------------------------------
            if not self._move_focus_stage(z_stage_name, target, f"attempt-{attempt}"):
                continue

            # --- focus ------------------------------------------------------
            focus_started = time.monotonic()
            try:
                core.fullFocus()
            except Exception:
                logger.debug(
                    f"Autofocus failed to engage with Z offset of {z_offset} um after "
                    f"{time.monotonic() - focus_started:.3f} s",
                    exc_info=True,
                )
                self._log_afc_state(f"fullFocus-error-attempt-{attempt}")
                continue
            logger.debug(
                f"Autofocus engaged with Z offset of {z_offset} um "
                f"(fullFocus took {time.monotonic() - focus_started:.3f} s, native MM "
                f"reference ~{NATIVE_FULL_FOCUS_S:.2f} s; "
                f"{time.monotonic() - afc_started:.3f} s total over {attempt} attempts)"
            )
            self._log_afc_state(f"fullFocus-success-attempt-{attempt}")
            return True

        # return z stage to original position if autofocus attempts failed
        logger.debug(
            f"All {len(z_offsets)} AFC attempts failed in "
            f"{time.monotonic() - afc_started:.3f} s; returning {z_stage_name} to "
            f"{z_position}"
        )
        self._move_focus_stage(z_stage_name, z_position, "restore-after-failure")

        logger.error(f"Autofocus call failed after {len(z_offsets)} attempts")
        self._log_afc_state("afc-failed")
        return False
