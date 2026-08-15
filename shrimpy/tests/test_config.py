"""Tests for shrimPy acquisition config validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pydantic import ValidationError
from useq import MDASequence

from shrimpy.config import ShrimpyMetadata, load_config

CONFIG_DIR = Path(__file__).parent.parent.parent / "config" / "mda" / "mantis"
DEMO_MDA_CONFIG = Path(__file__).parent / "artifacts" / "demo_mda_sequence.yaml"

METADATA = {
    "autofocus": {"enabled": True, "method": "demo-PFS", "stage": "Z"},
    "reset_hardware_sequencing_settings": [["Z", "UseSequences", "No"]],
}

CONFIG = {
    "channels": [{"config": "BF", "group": "Channel", "exposure": 10.0}],
    "z_plan": {"top": 1.0, "bottom": -1.0, "step": 0.5},
    "stage_positions": [{"x": 0, "y": 0}],
    "metadata": METADATA,
}


def _write(tmp_path: Path, obj: dict, name: str = "config.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(obj))
    return path


# ---------------------------------------------------------------------------
# ShrimpyMetadata
# ---------------------------------------------------------------------------


def test_sections_are_parsed():
    meta = ShrimpyMetadata.model_validate(METADATA)
    assert meta.autofocus.enabled is True
    assert meta.autofocus.method == "demo-PFS"
    assert meta.autofocus.stage == "Z"
    assert meta.reset_hardware_sequencing_settings == [("Z", "UseSequences", "No")]
    assert meta.dynatrack is None


def test_defaults_when_sections_omitted():
    meta = ShrimpyMetadata.from_sequence(MDASequence())
    assert meta.autofocus.enabled is False
    assert meta.autofocus.method is None
    assert meta.reset_hardware_sequencing_settings == []
    assert meta.dynatrack is None


def test_unknown_section_is_rejected():
    with pytest.raises(ValidationError):
        ShrimpyMetadata.model_validate({**METADATA, "autofocuss": {"enabled": True}})


def test_unknown_key_within_section_is_rejected():
    with pytest.raises(ValidationError):
        ShrimpyMetadata.model_validate({"autofocus": {"enabled": True, "methodd": "PFS"}})


def test_property_values_are_coerced_to_strings():
    meta = ShrimpyMetadata.model_validate(
        {"reset_hardware_sequencing_settings": [["XYStage", "MotorSpeedX-S(mm/s)", 5.75]]}
    )
    assert meta.reset_hardware_sequencing_settings == [
        ("XYStage", "MotorSpeedX-S(mm/s)", "5.75"),
    ]


def test_dynatrack_section_is_validated():
    meta = ShrimpyMetadata.model_validate(
        {
            **METADATA,
            "dynatrack": {
                "enabled": True,
                "input_channel": "BF",
                "tracking_channel": "BF",
            },
        }
    )
    assert meta.dynatrack is not None
    assert meta.dynatrack.input_channel == "BF"

    # A dynatrack section must be complete even when it is present but disabled
    with pytest.raises(ValidationError):
        ShrimpyMetadata.model_validate({"dynatrack": {"enabled": False}})


def test_from_sequence_reads_sequence_metadata():
    meta = ShrimpyMetadata.from_sequence(MDASequence(metadata=METADATA))
    assert meta.autofocus.stage == "Z"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_returns_sequence_with_metadata(tmp_path):
    sequence = load_config(_write(tmp_path, CONFIG))
    assert isinstance(sequence, MDASequence)
    assert sequence.channels[0].config == "BF"
    assert ShrimpyMetadata.from_sequence(sequence).autofocus.method == "demo-PFS"


def test_load_config_rejects_legacy_nesting(tmp_path):
    legacy = {**CONFIG, "metadata": {"mantis": METADATA}}
    with pytest.raises(ValueError, match="legacy config layout"):
        load_config(_write(tmp_path, legacy, "legacy.yaml"))


def test_load_config_rejects_invalid_section(tmp_path):
    bad = {**CONFIG, "metadata": {"autofocus": {"enabled": True, "stagee": "Z"}}}
    with pytest.raises(ValidationError):
        load_config(_write(tmp_path, bad, "bad.yaml"))


@pytest.mark.parametrize("config_path", sorted(CONFIG_DIR.glob("*.yaml")))
def test_shipped_configs_validate(config_path):
    load_config(config_path)


def test_demo_test_artifact_validates():
    sequence = load_config(DEMO_MDA_CONFIG)
    assert sequence.setup is not None
    assert ShrimpyMetadata.from_sequence(sequence).autofocus.method == "demo-PFS"
