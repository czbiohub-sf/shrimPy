from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mantis.analysis.AnalysisSettings import CharacterizeSettings
from mantis.cli import (  # Make sure the import path is correct
    _characterize_psf,
    characterize_psf,
)


def test_characterize_psf_deprecation():
    # Example inputs for the test
    zyx_data = np.random.rand(10, 10, 10)  # Random 3D data
    zyx_scale = (0.1, 0.1, 0.1)  # Example scale factors
    settings = CharacterizeSettings()  # Assuming default settings can be instantiated
    output_report_path = '/fake/path/to/report.html'
    input_dataset_path = '/fake/path/to/dataset'
    input_dataset_name = 'TestDataset'

    # Mock the necessary functions and methods
    with patch(
        'mantis.analysis.analyze_psf.detect_peaks', return_value=np.array([[1, 2, 3]])
    ) as mock_detect_peaks, patch(
        'mantis.analysis.analyze_psf.extract_beads',
        return_value=(np.random.rand(5, 5, 5, 5), [(0, 0, 0)]),
    ) as mock_extract_beads, patch(
        'mantis.analysis.analyze_psf.analyze_psf',
        return_value=(pd.DataFrame(), pd.DataFrame()),
    ) as mock_analyze_psf, patch(
        'mantis.analysis.analyze_psf.generate_report'
    ) as mock_generate_report, pytest.warns(
        DeprecationWarning
    ) as record:
        # Call the function
        _characterize_psf(
            zyx_data,
            zyx_scale,
            settings,
            output_report_path,
            input_dataset_path,
            input_dataset_name,
        )

    # Check if a deprecation warning was captured
    assert "biahub" in str(
        record.list[0].message
    ), "Deprecation warning for _characterize_psf not found or incorrect."

    # Optionally, you can check the calls to mocked functions to ensure they were invoked correctly
    mock_detect_peaks.assert_called_once()
    mock_extract_beads.assert_called_once()
    mock_analyze_psf.assert_called_once()
    mock_generate_report.assert_called_once()
