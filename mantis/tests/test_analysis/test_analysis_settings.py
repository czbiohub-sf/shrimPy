import numpy as np
import pytest

from pydantic.v1 import ValidationError

from mantis.analysis.AnalysisSettings import (
    DeskewSettings,
    RegistrationSettings,
    StabilizationSettings,
)


def test_deskew_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        DeskewSettings(
            pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=0.313, typo_param="test"
        )

    # Test negative value
    with pytest.raises(ValidationError):
        DeskewSettings(pixel_size_um=-3, ls_angle_deg=36, scan_step_um=0.313)

    # Test light sheet angle range
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=90, scan_step_um=0.313)

    # Test px_to_scan_ratio logic
    with pytest.raises(ValueError):
        DeskewSettings(pixel_size_um=0.116, ls_angle_deg=36, scan_step_um=None)


def test_example_deskew_settings(example_deskew_settings):
    _, settings = example_deskew_settings

    DeskewSettings(**settings)


def test_register_settings():
    # Test extra parameter
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong output shape size
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(4).tolist(),
            typo_param="test",
        )

    # Test wrong matrix shape
    with pytest.raises(ValidationError):
        RegistrationSettings(
            source_channel_index=0,
            target_channel_index=0,
            affine_transform_zyx=np.identity(5).tolist(),
            typo_param="test",
        )


def test_example_register_settings(example_register_settings):
    _, settings = example_register_settings
    RegistrationSettings(**settings)


def test_example_stabilize_timelapse_settings(example_stabilize_timelapse_settings):
    _, settings = example_stabilize_timelapse_settings
    StabilizationSettings(**settings)


from mantis.analysis.AnalysisSettings import (
    CharacterizeSettings,
    ConcatenateSettings,
    DeconvolveSettings,
    DeskewSettings,
    MyBaseModel,
    ProcessingSettings,
    PsfFromBeadsSettings,
    RegistrationSettings,
    StabilizationSettings,
    StitchSettings,
)


@pytest.mark.parametrize(
    "class_to_test, parameters",
    [
        (MyBaseModel, {}),
        (ProcessingSettings, {}),
        (PsfFromBeadsSettings, {}),
        (DeconvolveSettings, {}),
        (CharacterizeSettings, {}),
        (DeskewSettings, {"pixel_size_um": 0.1, "ls_angle_deg": 10.0, "scan_step_um": 0.02}),
        (
            RegistrationSettings,
            {
                "source_channel_names": ["channel1", "channel2"],
                "target_channel_name": "channel3",
                "affine_transform_zyx": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "keep_overhang": False,
                "time_indices": "all",
            },
        ),
        (
            ConcatenateSettings,
            {
                "concat_data_paths": ["path1", "path2"],
                "time_indices": "all",
                "channel_names": ["channel1", "channel2"],
                "X_slice": [0, 100],
                "Y_slice": [0, 100],
                "Z_slice": [0, 50],
                "chunks_czyx": [1, 512, 512, 3],
            },
        ),
        (
            StabilizationSettings,
            {
                "stabilization_estimation_channel": "channel1",
                "stabilization_type": "xyz",
                "stabilization_channels": ["channel1", "channel2"],
                "affine_transform_zyx_list": [
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                ],
                "time_indices": "all",
            },
        ),
        (
            StitchSettings,
            {
                "channels": ["channel1", "channel2"],
                "preprocessing": ProcessingSettings(),
                "postprocessing": ProcessingSettings(),
                "total_translation": {"x": [0.1, 0.1], "y": [0.1, 0.1]},
            },
        ),
    ],
)
def test_derror(class_to_test, parameters):
    with pytest.raises(ValueError):
        class_to_test(**parameters)