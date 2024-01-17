import numpy as np

from click.testing import CliRunner
from numpy import testing

from mantis.cli.apply_affine import rescale_voxel_size
from mantis.cli.main import cli


def test_apply_affine_cli(tmp_path, example_plate, example_apply_affine_settings):
    plate_path, _ = example_plate
    config_path, _ = example_apply_affine_settings
    output_path = tmp_path / "output.zarr"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "apply-affine",
            "-i",
            str(plate_path) + "/A/1/0",
            "-c",
            str(config_path),
            "-o",
            str(output_path),
        ],
        catch_exceptions=False,
    )

    assert output_path.exists()
    assert result.exit_code == 0


def test_apply_affine_to_scale():
    input = np.array([1, 1, 1])

    # Test real positive
    m1_diag = np.array([2, 3, 4])
    m1 = np.diag(m1_diag)
    output1 = rescale_voxel_size(m1, input)
    testing.assert_allclose(m1_diag, output1)

    # Test real with negative
    m2_diag = np.array([2, -3, 4])
    m2 = np.diag(m2_diag)
    output2 = rescale_voxel_size(m2, input)
    testing.assert_allclose(np.abs(m2_diag), output2)

    # Test transpose
    m3 = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 3]])
    output3 = rescale_voxel_size(m3, input)
    testing.assert_allclose(np.array([2, 1, 3]), output3)

    # Test rotation
    theta = np.pi / 3
    m4 = np.array(
        [
            [2, 0, 0],
            [0, 3 * np.cos(theta), -3 * np.sin(theta)],
            [0, 3 * np.sin(theta), 3 * np.cos(theta)],
        ]
    )
    output4 = rescale_voxel_size(m4, input)
    testing.assert_allclose(np.array([2, 3, 3]), output4)
