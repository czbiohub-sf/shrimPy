"""Tests for the ShrimpyConfig acquisition configuration model."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError
from useq import MDASequence

from shrimpy.config import SHRIMPY_METADATA_KEY, ShrimpyConfig

CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "mda" / "mantis"
DEMO_MDA_CONFIG = Path(__file__).parent / "artifacts" / "demo_mda_sequence.yaml"

MINIMAL = {
    "mda": {
        "channels": [{"config": "BF", "group": "Channel", "exposure": 10.0}],
        "z_plan": {"top": 1.0, "bottom": -1.0, "step": 0.5},
        "stage_positions": [{"x": 0, "y": 0}],
    },
    "autofocus": {"enabled": True, "method": "demo-PFS", "stage": "Z"},
    "mantis": {"reset_hardware_sequencing_settings": [["Z", "UseSequences", "No"]]},
}


def _write(tmp_path: Path, obj: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(obj))
    return path


# ---------------------------------------------------------------------------
# Structure and validation
# ---------------------------------------------------------------------------


def test_sections_are_parsed():
    config = ShrimpyConfig.model_validate(MINIMAL)
    assert isinstance(config.mda, MDASequence)
    assert config.mda.channels[0].config == "BF"
    assert config.autofocus.enabled is True
    assert config.autofocus.stage == "Z"
    assert config.mantis.reset_hardware_sequencing_settings == [
        ("Z", "UseSequences", "No"),
    ]
    assert config.dynatrack is None


def test_defaults_when_sections_omitted():
    config = ShrimpyConfig(mda=MDASequence())
    assert config.autofocus.enabled is False
    assert config.mantis.use_hardware_sequencing is True
    assert config.mantis.reset_hardware_sequencing_settings == []
    assert config.dynatrack is None


def test_unknown_top_level_section_is_rejected():
    with pytest.raises(ValidationError):
        ShrimpyConfig.model_validate({**MINIMAL, "isim": {"enabled": True}})


def test_unknown_key_within_section_is_rejected():
    with pytest.raises(ValidationError):
        ShrimpyConfig.model_validate(
            {**MINIMAL, "autofocus": {"enabled": True, "methodd": "PFS"}}
        )


def test_property_values_are_coerced_to_strings():
    config = ShrimpyConfig.model_validate(
        {
            "mda": {},
            "mantis": {
                "reset_hardware_sequencing_settings": [
                    ["XYStage", "MotorSpeedX-S(mm/s)", 5.75]
                ]
            },
        }
    )
    assert config.mantis.reset_hardware_sequencing_settings == [
        ("XYStage", "MotorSpeedX-S(mm/s)", "5.75"),
    ]


def test_dynatrack_section_is_validated():
    config = ShrimpyConfig.model_validate(
        {
            **MINIMAL,
            "dynatrack": {
                "enabled": True,
                "input_channel": "BF",
                "tracking_channel": "BF",
            },
        }
    )
    assert config.dynatrack is not None
    assert config.dynatrack.input_channel == "BF"

    # A dynatrack section must be complete even when it is present but disabled
    with pytest.raises(ValidationError):
        ShrimpyConfig.model_validate({**MINIMAL, "dynatrack": {"enabled": False}})


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def test_from_file_roundtrip(tmp_path):
    config = ShrimpyConfig.from_file(_write(tmp_path, MINIMAL))
    assert config.autofocus.method == "demo-PFS"

    out = tmp_path / "out.yaml"
    config.to_file(out)
    written = yaml.safe_load(out.read_text())
    assert set(written) == {"mda", "autofocus", "mantis"}
    assert ShrimpyConfig.from_file(out) == config


def test_from_file_rejects_legacy_layout(tmp_path):
    legacy = {
        "channels": MINIMAL["mda"]["channels"],
        "z_plan": MINIMAL["mda"]["z_plan"],
        "metadata": {"mantis": {"autofocus": {"enabled": False}}},
    }
    with pytest.raises(ValueError, match="legacy config layout"):
        ShrimpyConfig.from_file(_write(tmp_path, legacy, "legacy.yaml"))


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_shipped_configs_validate(config_path):
    ShrimpyConfig.from_file(config_path)


def test_demo_test_artifact_validates():
    config = ShrimpyConfig.from_file(DEMO_MDA_CONFIG)
    assert config.autofocus.method == "demo-PFS"
    assert config.mda.setup is not None


# ---------------------------------------------------------------------------
# Sequence round-trip (how the sections reach the engine)
# ---------------------------------------------------------------------------


def test_to_sequence_embeds_sections_in_metadata():
    config = ShrimpyConfig.model_validate(MINIMAL)
    sequence = config.to_sequence()
    sections = sequence.metadata[SHRIMPY_METADATA_KEY]
    assert sections["autofocus"]["method"] == "demo-PFS"
    assert "mda" not in sections


def test_from_sequence_recovers_sections():
    config = ShrimpyConfig.model_validate(MINIMAL)
    recovered = ShrimpyConfig.from_sequence(config.to_sequence())
    assert recovered.autofocus == config.autofocus
    assert recovered.mantis == config.mantis
    assert recovered.mda.channels == config.mda.channels


def test_to_sequence_preserves_other_metadata():
    config = ShrimpyConfig(mda=MDASequence(metadata={"pymmcore_widgets": {"version": "1"}}))
    sequence = config.to_sequence()
    assert sequence.metadata["pymmcore_widgets"] == {"version": "1"}
    assert SHRIMPY_METADATA_KEY in sequence.metadata


def test_from_sequence_defaults_for_plain_sequence():
    config = ShrimpyConfig.from_sequence(MDASequence())
    assert config.autofocus.enabled is False
    assert config.dynatrack is None


def test_from_sequence_rejects_non_mapping_section():
    with pytest.raises(ValueError, match="must be a mapping"):
        ShrimpyConfig.from_sequence(MDASequence(metadata={SHRIMPY_METADATA_KEY: ["nope"]}))
