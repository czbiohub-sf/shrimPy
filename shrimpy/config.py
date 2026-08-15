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

from pydantic import BaseModel, ConfigDict, Field, field_validator
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
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str | None = None
    stage: str | None = None


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
