"""Validated shrimPy acquisition configuration.

A shrimPy acquisition is described by a single YAML (or JSON) file with four
top-level sections::

    mda:          # useq.MDASequence: setup, stage_positions, time_plan, z_plan, channels
    autofocus:    # continuous-autofocus settings (see AutofocusConfig)
    mantis:       # microscope-specific hardware settings (see MicroscopeConfig)
    dynatrack:    # DynaTrack position tracking (see DynaTrackConfig)

Only ``mda`` is required. The file is validated by :class:`ShrimpyConfig`,
which rejects unknown keys (``extra="forbid"``) so a mistyped setting fails
before any hardware is touched::

    from shrimpy.config import ShrimpyConfig

    config = ShrimpyConfig.from_file("config/mda/mantis/demo.yaml")
    sequence = config.to_sequence()  # MDASequence carrying the other sections

Engines receive an ``MDASequence`` from the MDA runner, so the non-``mda``
sections travel with it under ``sequence.metadata['shrimpy']``
(:meth:`ShrimpyConfig.to_sequence`) and are read back with
:meth:`ShrimpyConfig.from_sequence`. This also persists them in the
acquisition's summary metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator
from useq import MDASequence

from shrimpy.dynatrack.tracking import DynaTrackConfig

__all__ = [
    "AutofocusConfig",
    "MicroscopeConfig",
    "ShrimpyConfig",
    "SHRIMPY_METADATA_KEY",
]

# Key under which the non-``mda`` config sections are carried in
# ``MDASequence.metadata``.
SHRIMPY_METADATA_KEY = "shrimpy"

# Top-level MDASequence fields; used to detect (and reject) legacy config files
# where the MDA settings were at the top level next to a ``metadata`` section.
_LEGACY_TOP_LEVEL_KEYS = frozenset(MDASequence.model_fields) - {"metadata"}


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


class AutofocusConfig(BaseModel):
    """Continuous-autofocus settings.

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


class MicroscopeConfig(BaseModel):
    """Microscope-specific hardware settings.

    Named after the microscope it configures in :class:`ShrimpyConfig` (today
    ``mantis:``; future scopes get their own section, e.g. ``isim:``).

    Hardware *setup* (ROI, imaging path, TriggerScope sequencing properties) is
    part of the acquisition itself and lives in ``mda.setup``; this section
    holds what the engine needs beyond the MDA sequence.

    Parameters
    ----------
    use_hardware_sequencing : bool
        Whether the engine drives the camera and DACs in sequenced (hardware
        triggered) mode.
    reset_hardware_sequencing_settings : list[tuple[str, str, str]]
        ``[device, property, value]`` triplets applied in
        ``teardown_sequence`` to return the hardware to a safe idle state
        (e.g. turning TTL blanking and DAC sequencing off).
    """

    model_config = ConfigDict(extra="forbid")

    use_hardware_sequencing: bool = True
    reset_hardware_sequencing_settings: list[tuple[str, str, str]] = Field(
        default_factory=list
    )

    _coerce_reset = field_validator("reset_hardware_sequencing_settings", mode="before")(
        _as_property_settings
    )


class ShrimpyConfig(BaseModel):
    """Top-level shrimPy acquisition configuration.

    Parameters
    ----------
    mda : MDASequence
        The multi-dimensional acquisition sequence (``setup``,
        ``stage_positions``, ``time_plan``, ``z_plan``, ``channels``, ...).
    autofocus : AutofocusConfig
        Continuous-autofocus settings. Defaults to disabled.
    mantis : MicroscopeConfig
        Mantis hardware settings. Defaults to hardware sequencing on with no
        reset properties. Other microscopes will get their own section (e.g.
        ``isim: MicroscopeConfig``).
    dynatrack : DynaTrackConfig | None
        DynaTrack position-tracking settings; ``None`` (the default) disables
        tracking.
    """

    model_config = ConfigDict(extra="forbid")

    mda: MDASequence
    autofocus: AutofocusConfig = Field(default_factory=AutofocusConfig)
    mantis: MicroscopeConfig = Field(default_factory=MicroscopeConfig)
    dynatrack: DynaTrackConfig | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> ShrimpyConfig:
        """Load and validate a config from a YAML or JSON file.

        Raises
        ------
        ValueError
            If the file uses the legacy layout (MDA fields at the top level
            with microscope settings nested under ``metadata``).
        """
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            obj = yaml.safe_load(path.read_bytes())
        elif path.suffix == ".json":
            import json

            obj = json.loads(path.read_bytes())
        else:  # pragma: no cover
            raise ValueError(f"Unknown file type: {path.suffix}")

        if not isinstance(obj, dict):
            raise ValueError(f"{path} does not contain a mapping of config sections.")
        if "mda" not in obj and _LEGACY_TOP_LEVEL_KEYS & set(obj):
            raise ValueError(
                f"{path} appears to use the legacy config layout. Nest the MDA "
                "settings (setup, stage_positions, time_plan, z_plan, channels, "
                "...) under a top-level 'mda:' key, and lift 'autofocus', "
                "'mantis' and 'dynatrack' out of 'metadata:' to the top level."
            )
        return cls.model_validate(obj)

    @classmethod
    def from_sequence(cls, sequence: MDASequence) -> ShrimpyConfig:
        """Rebuild a config from a sequence produced by :meth:`to_sequence`.

        Sections are read from ``sequence.metadata['shrimpy']``; missing
        sections fall back to their defaults (autofocus and DynaTrack
        disabled), so a plain ``MDASequence`` is valid input.
        """
        sections = (sequence.metadata or {}).get(SHRIMPY_METADATA_KEY) or {}
        if not isinstance(sections, dict):
            raise ValueError(
                f"sequence.metadata[{SHRIMPY_METADATA_KEY!r}] must be a mapping of "
                f"config sections, got {type(sections).__name__}."
            )
        return cls(mda=sequence, **sections)

    # -- serialization ------------------------------------------------------

    def sections(self) -> dict[str, Any]:
        """Return the non-``mda`` sections as JSON-serializable dicts."""
        return self.model_dump(mode="json", exclude={"mda"}, exclude_none=True)

    def to_sequence(self) -> MDASequence:
        """Return ``mda`` with the other sections in ``metadata['shrimpy']``."""
        metadata = dict(self.mda.metadata or {})
        metadata[SHRIMPY_METADATA_KEY] = self.sections()
        return self.mda.model_copy(update={"metadata": metadata})

    def to_file(self, path: str | Path) -> None:
        """Write the config to a YAML file with the standard section layout."""
        path = Path(path)
        obj = {"mda": self.mda.model_dump(mode="json", exclude_none=False), **self.sections()}
        with open(path, "w") as fh:
            yaml.safe_dump(obj, fh, default_flow_style=False, sort_keys=False)
