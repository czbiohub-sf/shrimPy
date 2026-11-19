"""Photomanipulation assembly: lasers steered by galvo mirrors, calibrated to a camera.

Vendored from coPylot ``copylot/assemblies/photom/photom.py`` (main @ 0bcc3a3),
BSD-3-Clause; see the package LICENSE.

Changes from upstream:

- the coPylot abstract camera/mirror base classes are replaced by the
  structural :class:`Camera` / :class:`Mirror` protocols below (only the
  members the assembly uses), so any object with those members works, coPylot
  devices included; lasers use :class:`copylot.hardware.lasers.abstract_laser.AbstractLaser`
- unused imports (``re.T``, vispy, matplotlib, the stage/DAQ bases) and tqdm
  are dropped; ``print`` is now logging
- scikit-image, tifffile and matplotlib are imported only by
  :meth:`PhotomAssembly.calibrate_w_camera` (``calibration`` extra), so moving the
  mirrors needs only the core dependencies
- ``calibrate_w_camera``: accepts a ``str`` ``config_file`` (upstream called
  ``Path.endswith``), and returns ``plot_save_path=None`` instead of raising
  ``NameError`` when ``verbose`` is off
- ``capture()`` raises ``NotImplementedError`` (upstream silently returned ``None``)
- ``calibrate_w_camera`` names the frame with no detectable spot (upstream
  failed with a numpy broadcasting error)
"""

from __future__ import annotations

import logging
import time

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from warnings import warn

import numpy as np

from copylot.assemblies.photom.utils.affine_transform import AffineTransform
from copylot.assemblies.photom.utils.scanning_algorithms import generate_grid_points
from copylot.hardware.lasers.abstract_laser import AbstractLaser

logger = logging.getLogger(__name__)

#: ``(width, height, offset_x, offset_y)`` assumed when no camera is given.
DEFAULT_SENSOR_SIZE = (2048, 2448, 0, 0)


@runtime_checkable
class Mirror(Protocol):
    """What :class:`PhotomAssembly` needs from a galvo mirror."""

    #: Set by :meth:`PhotomAssembly.init_mirrors`.
    affine_transform_obj: AffineTransform | None

    @property
    def position(self) -> Sequence[float]: ...

    @position.setter
    def position(self, value: Sequence[float]) -> None: ...


@runtime_checkable
class Camera(Protocol):
    """What :class:`PhotomAssembly` needs from a camera."""

    @property
    def image_size(self) -> tuple[int, int, int, int]:
        """``(width, height, offset_x, offset_y)``."""
        ...

    def snap(self) -> np.ndarray: ...


class PhotomAssembly:
    """Lasers and galvo mirrors, with one camera-to-mirror affine per mirror.

    Parameters
    ----------
    laser : list[AbstractLaser]
        Lasers, addressed by index.
    mirror : list[Mirror]
        Galvo mirrors, addressed by index.
    affine_matrix_path : list[Path]
        One affine-calibration YAML per mirror (see
        :class:`~copylot.assemblies.photom.utils.affine_transform.AffineTransform`).
    camera : list[Camera], optional
        Cameras used for calibration; the first sets :attr:`sensor_size`.
    dac : list, optional
        DAQ-driven mirrors are not implemented; passing one makes the mirror
        position methods raise ``NotImplementedError``.
    """

    def __init__(
        self,
        laser: list[AbstractLaser],
        mirror: list[Mirror],
        affine_matrix_path: list[Path],
        camera: list[Camera] | None = None,
        dac: list[Any] | None = None,
    ):
        self.camera = camera
        self.laser = laser
        self.mirror = mirror
        self.DAC = dac
        self.affine_matrix_path = affine_matrix_path

        if self.camera and len(self.camera) > 0:
            self.sensor_size = self.camera[0].image_size
            logger.debug(
                f"Sensor size: {self.sensor_size} (width, height, offset_x, offset_y)"
            )
        else:
            warn(
                f"No camera provided. Using default size {DEFAULT_SENSOR_SIZE[:2]}.",
                stacklevel=2,
            )
            self.sensor_size = DEFAULT_SENSOR_SIZE
        self.init_mirrors()

    def init_mirrors(self):
        """Load each mirror's affine calibration onto ``mirror.affine_transform_obj``."""
        if len(self.mirror) != len(self.affine_matrix_path):
            raise ValueError(
                f"Got {len(self.mirror)} mirrors but {len(self.affine_matrix_path)} "
                "affine matrix paths; pass one per mirror"
            )
        for mirror, path in zip(self.mirror, self.affine_matrix_path, strict=True):
            mirror.affine_transform_obj = AffineTransform(config_file=path)

    ## Camera functions
    def capture(self, camera_index: int, exposure_time: float) -> list:
        raise NotImplementedError("PhotomAssembly.capture is not implemented")

    ## Mirror functions
    def _check_mirror_index(self, mirror_index: int) -> None:
        if not 0 <= mirror_index < len(self.mirror):
            raise IndexError("Mirror index out of range.")
        if self.DAC is not None:
            raise NotImplementedError("DAQ-driven mirrors are not implemented.")

    def get_position(self, mirror_index: int) -> list[float]:
        """Current mirror position, in mirror coordinates."""
        self._check_mirror_index(mirror_index)
        return list(self.mirror[mirror_index].position)

    def set_position(self, mirror_index: int, position: list[float]):
        """Point a mirror at ``position`` (x, y) in camera coordinates.

        The position is mapped to mirror coordinates through the mirror's
        affine calibration.
        """
        self._check_mirror_index(mirror_index)
        mirror = self.mirror[mirror_index]
        new_position = mirror.affine_transform_obj.apply_affine(position)
        logger.debug(f"Mirror {mirror_index}: camera {position} -> mirror {new_position}")
        mirror.position = [new_position[0][0], new_position[1][0]]

    def calibrate_w_camera(
        self,
        mirror_index: int,
        camera_index: int,
        rectangle_boundaries: tuple[tuple[float, float], tuple[float, float]],
        grid_n_points: int = 5,
        config_file: Path | str | None = None,
        save_calib_stack_path: Path | str | None = None,
        verbose: bool = False,
    ) -> tuple[np.ndarray, Path | None]:
        """Calibrate a mirror's camera-to-mirror affine from a grid of laser spots.

        Steps the mirror over a grid spanning ``rectangle_boundaries`` (mirror
        coordinates), snaps a frame at each point, finds the spot, and fits the
        affine mapping camera (x, y) to mirror (x, y). The fit is stored on the
        mirror's ``affine_transform_obj``.

        Parameters
        ----------
        mirror_index, camera_index : int
            Which mirror and camera to use.
        rectangle_boundaries : ((start_x, start_y), (end_x, end_y))
            Mirror-coordinate extent of the calibration grid.
        grid_n_points : int
            Grid points per row and per column.
        config_file : Path, optional
            ``.yml``/``.yaml`` to save the fitted matrix to.
        save_calib_stack_path : Path, optional
            Directory to save the calibration image stack (TIFF) to.
        verbose : bool
            Also save a centroid plot and the grid/peak points (``.npz``), to
            ``save_calib_stack_path`` or the working directory.

        Returns
        -------
        T_affine : np.ndarray
            The fitted 3x3 matrix.
        plot_save_path : Path or None
            The centroid plot, if ``verbose``.
        """
        from copylot.assemblies.photom.utils import image_analysis as ia

        if self.camera is None:
            raise ValueError("Calibration requires a camera")
        if not 0 <= mirror_index < len(self.mirror):
            raise IndexError("Mirror index out of range.")
        if not 0 <= camera_index < len(self.camera):
            raise IndexError("Camera index out of range.")

        current_width, current_height, _, _ = self.camera[camera_index].image_size
        logger.info(
            f"Calibrating mirror_idx <{mirror_index}> with camera_idx <{camera_index}>"
        )
        grid_points = generate_grid_points(
            rectangle_size=rectangle_boundaries, n_points=grid_n_points
        )

        img_sequence = np.zeros(
            (len(grid_points), current_height, current_width), dtype="uint16"
        )
        for idx, coord in enumerate(grid_points):
            logger.debug(f"Calibration point {idx + 1}/{len(grid_points)}: {coord}")
            self.mirror[mirror_index].position = [coord[0], coord[1]]
            img_sequence[idx] = self.camera[camera_index].snap()

        # Images are (y, x); the affine is fit in (x, y).
        peak_coords = np.zeros((len(grid_points), 2))
        for idx, img in enumerate(img_sequence):
            peaks = ia.find_objects_centroids(
                img, sigma=5, threshold_rel=1.0, min_distance=30, max_num_peaks=1
            )
            if len(peaks) == 0:
                raise RuntimeError(
                    f"No laser spot found in calibration frame {idx} (mirror at "
                    f"{list(grid_points[idx])}); the spot must be on the sensor and at "
                    "least 30 px from its edge. Check the laser and rectangle_boundaries."
                )
            peak_coords[idx] = peaks[0]
        peak_coords_xy = peak_coords[:, [1, 0]]

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if save_calib_stack_path is not None:
            import tifffile

            save_calib_stack_path = Path(save_calib_stack_path)
            save_calib_stack_path.mkdir(parents=True, exist_ok=True)
            output_path_name = save_calib_stack_path / f"calibration_images_{timestamp}.tif"
            logger.info(f"Saving calibration stack to {output_path_name}")
            tifffile.imwrite(output_path_name, img_sequence, dtype="uint16", imagej=True)

        plot_save_path = None
        if verbose:
            out_dir = Path(save_calib_stack_path) if save_calib_stack_path else Path.cwd()
            plot_save_path = out_dir / f"calibration_plot_{timestamp}.png"
            logger.info(f"Saving coordinate and image stack plot to {plot_save_path}")
            ia.plot_centroids(img_sequence, peak_coords, mip=True, save_path=plot_save_path)
            np.savez(
                out_dir / f"calibration_points_{timestamp}.npz",
                grid_points=np.asarray(grid_points),
                peak_coords_xy=np.asarray(peak_coords_xy),
            )

        affine = self.mirror[mirror_index].affine_transform_obj
        T_affine = affine.compute_affine_matrix(peak_coords_xy, grid_points)
        logger.info(f"Affine matrix: {T_affine}")

        if config_file is not None:
            config_file = Path(config_file)
            if config_file.suffix not in (".yml", ".yaml"):
                raise ValueError(f"config_file must be .yml or .yaml, got {config_file}")
            affine.save_matrix(matrix=T_affine, config_file=config_file)
        return T_affine, plot_save_path

    ## Laser functions
    def get_laser_power(self, laser_index: int) -> float:
        return self.laser[laser_index].power

    def set_laser_power(self, laser_index: int, power: float):
        self.laser[laser_index].power = power

    ## Conversion between image coordinates and mirror coordinates or DAC voltages
    def normalize(self, value, min_val, max_val):
        if max_val == min_val:
            raise ValueError("Maximum and minimum values cannot be the same")
        return (value - min_val) / (max_val - min_val)

    def convert_values(
        self, input_values: list, input_range: list, output_range: list
    ) -> list:
        """Linearly map values from ``input_range`` to ``output_range``.

        Parameters
        ----------
        input_values : list or float
            Values to convert.
        input_range, output_range : list
            ``[min, max]`` of each range.
        """
        if not (len(input_range) == 2 and len(output_range) == 2):
            raise ValueError("Input and output ranges must each contain exactly two elements")
        if not (input_range[0] < input_range[1] and output_range[0] < output_range[1]):
            raise ValueError("In both ranges, the first element must be less than the second")

        if not isinstance(input_values, list):
            input_values = [input_values]

        input_min, input_max = min(input_range), max(input_range)
        output_min, output_max = min(output_range), max(output_range)
        output_span = output_max - output_min

        return [
            self.normalize(v, input_min, input_max) * output_span + output_min
            for v in input_values
        ]
