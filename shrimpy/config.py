"""Validation of shrimPy acquisition configuration files.

An acquisition config file *is* a ``useq.MDASequence``: ``setup``,
``stage_positions``, ``time_plan``, ``z_plan``, ``channels``, ... at the top
level, with the microscope settings folded directly into ``metadata``::

    setup: ...
    stage_positions: ...
    channels: ...
    metadata:
      autofocus: {enabled: true, method: PFS, stage: ZDrive}
      reset_hardware_sequencing_settings:
        - ['TS2_DAC03', 'Sequence', 'Off']
      dynatrack: {enabled: true, input_channel: BF, tracking_channel: BF}

:class:`ShrimpyMetadata` validates the ``metadata`` sections that shrimPy
itself consumes, so a mistyped setting fails before any hardware is touched::

    sequence = load_config("config/mda/mantis/demo.yaml")  # validates metadata
    meta = ShrimpyMetadata.from_sequence(sequence)  # engines read this

Validation is strict throughout (``extra="forbid"``): an unrecognized metadata
section, or an unknown key within one, is an error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from useq import MDASequence

from shrimpy.dynatrack.tracking import DynaTrackConfig

__all__ = [
    "AutofocusSettings",
    "ShrimpyMetadata",
    "load_config",
]


def _as_property_settings(value: Any) -> Any:
    """Coerce ``[device, property, value]`` triplets to all-string tuples.

    YAML property values are often written unquoted (``5.75``, ``2048``), but
    Micro-Manager property values are strings. Coerce here so configs stay
    forgiving while the model type stays strict.
    """
    if isinstance(value, (list, tuple)):
        return [
            [str(item) for item in setting] if isinstance(setting, (list, tuple)) else setting
            for setting in value
        ]
    return value


class AutofocusSettings(BaseModel):
    """Continuous-autofocus settings, from ``metadata.autofocus``.

    Parameters
    ----------
    enabled : bool
        Master switch. When False the engine never engages autofocus.
    method : str | None
        Autofocus device / method name, e.g. ``"PFS"`` for the Nikon Perfect
        Focus System or ``"demo-PFS"`` for the simulated method used with the
        Micro-Manager demo config.
    stage : str | None
        Name of the Z stage that is moved to help engage autofocus (e.g.
        ``"ZDrive"``). Z positions are not written to this stage while
        autofocus is enabled, so the focus lock is not disturbed.
    home_focus_device : bool
        Return the Core-Focus device to the position it held before the run
        (its "home") before engaging autofocus at each new position, and once
        at the end of the sequence.

        Only set this when the Core-Focus device's position *changes the plane
        autofocus locks onto*. On the Dragonfly it does: the z_plan rides the
        ASI piezo while AFC drives the Leica FocusDrive, so without homing, AFC
        locks onto wherever the previous Z-stack stopped — the top slice — and
        the whole stack is acquired below focus.

        On mantis the Core-Focus device is the light-sheet scan galvo, which
        moves neither the sample nor the objective, so PFS on ``ZDrive`` is
        indifferent to it; homing would only add a redundant galvo move before
        every hardware-sequenced burst. Hence the default of False: the
        Core-Focus device differing from ``stage`` does not by itself mean the
        two interact.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str | None = None
    stage: str | None = None
    home_focus_device: bool = False


class ShrimpyMetadata(BaseModel):
    """The shrimPy sections of ``MDASequence.metadata``.

    Unknown sections are rejected (``extra="forbid"``), so a mistyped setting
    fails before any hardware is touched.

    Parameters
    ----------
    autofocus : AutofocusSettings
        Continuous-autofocus settings. Defaults to disabled.
    reset_hardware_sequencing_settings : list[tuple[str, str, str]]
        ``[device, property, value]`` triplets applied in
        ``teardown_sequence`` to return the hardware to a safe idle state
        (e.g. turning TTL blanking and DAC sequencing off).
    dynatrack : DynaTrackConfig | None
        DynaTrack position-tracking settings; ``None`` (the default) disables
        tracking. A section that is present is validated even when
        ``enabled: false``, so ``input_channel`` / ``tracking_channel`` are
        required; omit the section entirely to disable tracking.

    Raises
    ------
    pydantic.ValidationError
        If autofocus and DynaTrack are both enabled: both correct Z, so
        together they fight over the focal plane (see
        :meth:`_check_autofocus_dynatrack_exclusive`).
    """

    model_config = ConfigDict(extra="forbid")

    autofocus: AutofocusSettings = Field(default_factory=AutofocusSettings)
    reset_hardware_sequencing_settings: list[tuple[str, str, str]] = Field(
        default_factory=list
    )
    dynatrack: DynaTrackConfig | None = None

    _coerce_reset = field_validator("reset_hardware_sequencing_settings", mode="before")(
        _as_property_settings
    )

    @model_validator(mode="after")
    def _check_autofocus_dynatrack_exclusive(self) -> ShrimpyMetadata:
        """Reject configs that enable both autofocus and DynaTrack.

        Both features drive Z: continuous autofocus holds the focal plane
        against a reference surface, while DynaTrack writes a corrected Z onto
        each event to follow the sample. Running them together means they
        fight over the focal plane, and the correction DynaTrack writes onto a
        sequenced event is not the position autofocus engages at.
        """
        if self.autofocus.enabled and self.dynatrack is not None and self.dynatrack.enabled:
            raise ValueError(
                "autofocus and dynatrack cannot both be enabled: both correct Z "
                "and would fight over the focal plane. Disable one of them "
                "(set 'autofocus.enabled: false' or 'dynatrack.enabled: false')."
            )
        return self

    @classmethod
    def from_sequence(cls, sequence: MDASequence) -> ShrimpyMetadata:
        """Validate and return the shrimPy sections of a sequence's metadata.

        A sequence with no metadata yields all-default settings (autofocus and
        DynaTrack disabled), so a plain ``MDASequence`` is valid input.
        """
        metadata = sequence.metadata or {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"sequence.metadata must be a mapping, got {type(metadata).__name__}."
            )
        return cls.model_validate(metadata)


def load_config(path: str | Path) -> MDASequence:
    """Load an acquisition config file and validate its shrimPy metadata.

    Parameters
    ----------
    path : str | Path
        Path to a YAML or JSON acquisition config (an ``MDASequence``, with
        shrimPy settings under ``metadata``).

    Returns
    -------
    MDASequence
        The parsed sequence. Its ``metadata`` is left untouched; use
        :meth:`ShrimpyMetadata.from_sequence` to read the validated sections.

    Raises
    ------
    ValueError
        If the file uses the legacy layout, where the microscope settings were
        nested one level deeper under ``metadata.mantis``.
    pydantic.ValidationError
        If the sequence or its shrimPy metadata sections are invalid.
    """
    sequence = MDASequence.from_file(path)
    metadata = sequence.metadata or {}
    if isinstance(metadata, dict) and isinstance(metadata.get("mantis"), dict):
        raise ValueError(
            f"{path} appears to use the legacy config layout: the microscope "
            "settings under 'metadata.mantis' are now folded directly into "
            "'metadata' (autofocus, reset_hardware_sequencing_settings, dynatrack)."
        )
    ShrimpyMetadata.from_sequence(sequence)
    return sequence
