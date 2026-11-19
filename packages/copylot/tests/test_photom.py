"""PhotomAssembly and its affine calibration, with fake mirror/camera/laser."""

from __future__ import annotations

import numpy as np
import pytest

from copylot.assemblies.photom.photom import Camera, Mirror, PhotomAssembly
from copylot.assemblies.photom.utils.affine_transform import (
    AffineTransform,
    AffineTransformationSettings,
)
from copylot.assemblies.photom.utils.scanning_algorithms import (
    calculate_rectangle_corners,
    generate_grid_points,
)

# camera (x, y) = SCALE * mirror (x, y) + OFFSET
SCALE = 2.0
OFFSET = np.array([20.0, 30.0])


class FakeMirror:
    def __init__(self):
        self.affine_transform_obj = None
        self._position = [0.0, 0.0]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = [float(value[0]), float(value[1])]


class FakeCamera:
    """Images a Gaussian laser spot wherever the mirror points."""

    def __init__(self, mirror, width=240, height=200):
        self.mirror = mirror
        self.width, self.height = width, height

    @property
    def image_size(self):
        return (self.width, self.height, 0, 0)

    def snap(self):
        x, y = SCALE * np.asarray(self.mirror.position) + OFFSET
        yy, xx = np.mgrid[: self.height, : self.width]
        spot = 1000 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 3.0**2))
        return (spot + 10).astype(np.uint16)


class FakeLaser:
    power = 0.0


@pytest.fixture
def identity_yaml(tmp_path):
    path = tmp_path / "affine.yml"
    AffineTransform().save_matrix(config_file=path)
    return path


def test_fakes_satisfy_protocols():
    mirror = FakeMirror()
    assert isinstance(mirror, Mirror)
    assert isinstance(FakeCamera(mirror), Camera)


def test_affine_transform_without_config_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    affine = AffineTransform()
    np.testing.assert_array_equal(affine.get_affine_matrix(), np.eye(3))
    assert list(tmp_path.iterdir()) == []


def test_affine_transform_round_trip(tmp_path):
    matrix = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [0.0, 0.0, 1.0]])
    path = tmp_path / "affine.yml"
    AffineTransform(tx_matrix=matrix).save_matrix(config_file=path)

    loaded = AffineTransform(config_file=path)
    np.testing.assert_array_equal(loaded.get_affine_matrix(), matrix)
    # channel-wise in, channel-wise out (plus the homogeneous row)
    assert loaded.apply_affine([1.0, 2.0]) == [[3.0], [5.0], [1.0]]


def test_affine_settings_reject_non_3x3_and_extra_keys():
    with pytest.raises(ValueError):
        AffineTransformationSettings(affine_transform_yx=[[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        AffineTransformationSettings(affine_transform_yx=np.eye(3).tolist(), typo=1)


def test_grid_and_corners():
    grid = generate_grid_points([[-1, -1], [1, 1]], 3)
    assert grid.shape == (9, 2)
    np.testing.assert_array_equal(grid[[0, 1, 3, 8]], [[-1, -1], [0, -1], [-1, 0], [1, 1]])
    assert calculate_rectangle_corners((2, 4), center=(1, 1)) == [
        [0, -1],
        [2, -1],
        [2, 3],
        [0, 3],
    ]


def test_set_position_applies_mirror_affine(tmp_path):
    path = tmp_path / "affine.yml"
    AffineTransform(tx_matrix=[[0.5, 0, -10], [0, 0.5, -15], [0, 0, 1]]).save_matrix(
        config_file=path
    )
    mirror = FakeMirror()
    photom = PhotomAssembly(
        laser=[FakeLaser()], mirror=[mirror], affine_matrix_path=[path], camera=[]
    )

    photom.set_position(0, [40.0, 50.0])
    assert photom.get_position(0) == [10.0, 10.0]

    photom.set_laser_power(0, 3.5)
    assert photom.get_laser_power(0) == 3.5


def test_mirror_and_matrix_count_must_match(identity_yaml):
    with pytest.raises(ValueError, match="one per mirror"):
        PhotomAssembly(
            laser=[], mirror=[FakeMirror(), FakeMirror()], affine_matrix_path=[identity_yaml]
        )


def test_convert_values(identity_yaml):
    photom = PhotomAssembly(
        laser=[], mirror=[FakeMirror()], affine_matrix_path=[identity_yaml]
    )
    assert photom.convert_values([0, 5, 10], [0, 10], [-1, 1]) == [-1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        photom.convert_values([1], [10, 0], [0, 1])


def test_calibrate_w_camera_recovers_camera_to_mirror_affine(identity_yaml, tmp_path):
    pytest.importorskip("skimage")
    pytest.importorskip("tifffile")
    pytest.importorskip("matplotlib")

    mirror = FakeMirror()
    camera = FakeCamera(mirror)
    photom = PhotomAssembly(
        laser=[], mirror=[mirror], affine_matrix_path=[identity_yaml], camera=[camera]
    )
    assert photom.sensor_size == (240, 200, 0, 0)

    out_yaml = tmp_path / "calibrated.yml"
    T_affine, plot_path = photom.calibrate_w_camera(
        mirror_index=0,
        camera_index=0,
        rectangle_boundaries=((10, 10), (70, 50)),
        grid_n_points=3,
        config_file=str(out_yaml),
        save_calib_stack_path=tmp_path / "calib",
        verbose=True,
    )

    # mirror = (camera - OFFSET) / SCALE
    expected = np.array(
        [[1 / SCALE, 0, -OFFSET[0] / SCALE], [0, 1 / SCALE, -OFFSET[1] / SCALE], [0, 0, 1]]
    )
    np.testing.assert_allclose(T_affine, expected, atol=0.05)
    np.testing.assert_allclose(AffineTransform(config_file=out_yaml).T_affine, T_affine)
    assert plot_path is not None and plot_path.exists()
    assert len(list((tmp_path / "calib").glob("calibration_images_*.tif"))) == 1

    # pointing at a camera pixel now lands the spot there
    photom.set_position(0, [100.0, 90.0])
    np.testing.assert_allclose(
        SCALE * np.asarray(mirror.position) + OFFSET, [100.0, 90.0], atol=0.5
    )


def test_calibrate_w_camera_reports_missing_spot(identity_yaml):
    pytest.importorskip("skimage")

    class DarkCamera:
        image_size = (240, 200, 0, 0)

        def snap(self):
            return np.full((200, 240), 10, dtype=np.uint16)

    photom = PhotomAssembly(
        laser=[],
        mirror=[FakeMirror()],
        affine_matrix_path=[identity_yaml],
        camera=[DarkCamera()],
    )
    with pytest.raises(RuntimeError, match="No laser spot found in calibration frame 0"):
        photom.calibrate_w_camera(0, 0, ((10, 10), (70, 50)), grid_n_points=2)
