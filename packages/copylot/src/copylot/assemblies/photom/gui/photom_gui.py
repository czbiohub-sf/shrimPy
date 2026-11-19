"""Launch the photom GUI from a YAML config, with real or simulated devices.

Vendored from coPylot ``copylot/assemblies/photom/gui/photom_gui.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE.

Upstream was a script: ``DEMO_MODE``, the config path, the camera sensor size
and the Arduino port (``COM10``) were constants edited in the file. Here they
are a validated config (:class:`PhotomConfig`) plus a ``demo`` flag, and
:func:`launch` runs the app. The config keeps upstream's keys, so existing
photom configs (``demo/photom_VIS_config.yml``) load unchanged.

Run it with ``shrimpy gui photom`` or ``python -m copylot.assemblies.photom.gui``.
"""

from __future__ import annotations

import logging
import sys
import warnings

from collections.abc import Callable
from pathlib import Path

import yaml

from pydantic import BaseModel, ConfigDict, Field

from copylot.assemblies.photom.photom import PhotomAssembly

logger = logging.getLogger(__name__)


class _Model(BaseModel):
    # extra="forbid" so a typo in the config fails loudly instead of being ignored
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class LaserConfig(_Model):
    name: str
    com_port: str | None = Field(default=None, alias="COM_port")
    serial_number: str | None = None


class MirrorConfig(_Model):
    name: str
    com_port: str | None = Field(default=None, alias="COM_port")
    x_position: float = 0.0
    y_position: float = 0.0
    #: Camera-to-mirror calibration YAML; identity (uncalibrated) if omitted.
    affine_matrix_path: Path | None = None


class ArduinoConfig(_Model):
    com_port: str = Field(alias="COM_port")
    baud_rate: int = 115200


class CalibrationConfig(_Model):
    #: ``[[x0, y0], [x1, y1]]`` mirror-coordinate extent of the calibration grid.
    mirror_roi: tuple[tuple[float, float], tuple[float, float]] = (
        (-0.01, -0.03),
        (0.015, 0.0),
    )
    grid_n_points: int = Field(default=5, ge=2)
    #: Where calibration stacks, plots and points are written (default: working dir).
    save_dir: Path | None = None


class PhotomConfig(_Model):
    lasers: list[LaserConfig] = Field(min_length=1)
    mirrors: list[MirrorConfig] = Field(min_length=1)
    #: Arduino PWM controllers; upstream always opened one on COM10.
    arduinos: list[ArduinoConfig] = Field(default_factory=list)
    #: Camera sensor (height, width) the laser-marker window is scaled to.
    camera_sensor_size_yx: tuple[int, int] = (2048, 2448)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PhotomConfig:
        """Load a config; relative paths in it resolve against the config's directory."""
        path = Path(path)
        with open(path) as f:
            config = cls.model_validate(yaml.safe_load(f))
        base = path.parent
        for mirror in config.mirrors:
            if (
                mirror.affine_matrix_path is not None
                and not mirror.affine_matrix_path.is_absolute()
            ):
                mirror.affine_matrix_path = base / mirror.affine_matrix_path
        save_dir = config.calibration.save_dir
        if save_dir is not None and not save_dir.is_absolute():
            config.calibration.save_dir = base / save_dir
        return config


#: Two lasers and one mirror, for ``--demo`` without a config.
DEMO_CONFIG = PhotomConfig(
    lasers=[LaserConfig(name="laser_405"), LaserConfig(name="laser_488")],
    mirrors=[MirrorConfig(name="mirror_1_VIS")],
    arduinos=[ArduinoConfig(COM_port="COM_DEMO")],
)


def make_photom_assembly(config: PhotomConfig, demo: bool = False) -> PhotomAssembly:
    """Connect the configured lasers and mirrors (or their mocks) into an assembly."""
    if demo:
        from copylot.assemblies.photom.photom_mock_devices import MockLaser as Laser
        from copylot.assemblies.photom.photom_mock_devices import MockMirror as Mirror
    else:
        from copylot.hardware.lasers.vortran.vortran import VortranLaser as Laser
        from copylot.hardware.mirrors.optotune.mirror import OptoMirror as Mirror

    lasers = [
        Laser(name=laser.name, port=laser.com_port, serial_number=laser.serial_number)
        for laser in config.lasers
    ]
    mirrors = [
        Mirror(
            name=mirror.name,
            com_port=mirror.com_port,
            pos_x=mirror.x_position,
            pos_y=mirror.y_position,
        )
        for mirror in config.mirrors
    ]
    with warnings.catch_warnings():
        # The GUI attaches a camera only while calibrating; the assembly's
        # "no camera, default sensor size" warning does not apply (the GUI scales
        # to config.camera_sensor_size_yx).
        warnings.filterwarnings("ignore", message="No camera provided")
        return PhotomAssembly(
            laser=lasers,
            mirror=mirrors,
            affine_matrix_path=[mirror.affine_matrix_path for mirror in config.mirrors],
            camera=[],
        )


def make_arduinos(config: PhotomConfig, demo: bool = False) -> list:
    if demo:
        from copylot.assemblies.photom.photom_mock_devices import MockArduinoPWM as ArduinoPWM
    else:
        from copylot.assemblies.photom.utils.arduino import ArduinoPWM
    return [ArduinoPWM(serial_port=a.com_port, baud_rate=a.baud_rate) for a in config.arduinos]


def demo_camera_factory(photom_assembly: PhotomAssembly, config: PhotomConfig) -> Callable:
    """A factory for a simulated camera imaging the laser spot of mirror 0.

    The calibration ROI is imaged across the middle 60% of the sensor (in its
    limiting dimension), clear of the border that spot detection excludes, so
    demo calibration recovers a well-conditioned camera-to-mirror affine.
    """
    from copylot.assemblies.photom.photom_mock_devices import MockCamera

    (x0, y0), (x1, y1) = config.calibration.mirror_roi
    height, width = config.camera_sensor_size_yx
    pixels_per_unit = 0.6 * min(width / abs(x1 - x0), height / abs(y1 - y0))

    def factory():
        cam = MockCamera(
            photom_assembly.mirror[0],
            sensor_size_yx=(height, width),
            mirror_center=((x0 + x1) / 2, (y0 + y1) / 2),
            pixels_per_unit=pixels_per_unit,
        )
        cam.open()
        return cam

    return factory


def check_calibration_files(
    config: PhotomConfig, calibration_files: list[Path | None]
) -> list[Path | None]:
    """Validate calibration files against the config before any hardware is opened.

    Raises
    ------
    ValueError
        More files than mirrors, or a file that is not a valid calibration
        (a 3x3 ``affine_transform_yx``).
    FileNotFoundError
        A file that does not exist.
    """
    from copylot.assemblies.photom.utils.affine_transform import AffineTransform

    if len(calibration_files) > len(config.mirrors):
        names = ", ".join(m.name for m in config.mirrors)
        raise ValueError(
            f"Got {len(calibration_files)} calibration files but the config has "
            f"{len(config.mirrors)} mirror(s) ({names}); pass at most one per mirror, "
            "in mirror order."
        )
    for path in calibration_files:
        if path is not None:
            AffineTransform(config_file=path)  # raises on a missing or malformed file
    return list(calibration_files)


def launch(
    config: PhotomConfig | None = None,
    demo: bool = False,
    camera_factory: Callable | None = None,
    calibration_files: list[Path | None] | None = None,
) -> int:
    """Build the devices, show the photom GUI and run the Qt event loop.

    Parameters
    ----------
    config : PhotomConfig, optional
        Devices and calibration settings. Defaults to :data:`DEMO_CONFIG` when
        ``demo`` is set; required otherwise.
    demo : bool
        Use simulated lasers, mirrors, Arduino and calibration camera.
    camera_factory : callable, optional
        Returns an opened camera for calibration. Defaults to the simulated
        camera in demo mode; without one, the Calibrate button is disabled.
    calibration_files : list of (Path or None), optional
        Camera-to-mirror calibration per mirror, in mirror order, loaded as if by
        the Load Calibration button. Overrides the mirror's ``affine_matrix_path``;
        ``None`` (or a shorter list) keeps it.

    Returns
    -------
    int
        The Qt event loop's exit code.
    """
    from qtpy.QtWidgets import QApplication

    from copylot.assemblies.photom.gui.windows import CalibrationSettings, PhotomApp

    if config is None:
        if not demo:
            raise ValueError("A photom config is required unless demo=True")
        config = DEMO_CONFIG
    calibration_files = check_calibration_files(config, calibration_files or [])

    app = QApplication.instance() or QApplication(sys.argv)

    photom_assembly = make_photom_assembly(config, demo=demo)
    arduinos = make_arduinos(config, demo=demo)
    if camera_factory is None and demo:
        camera_factory = demo_camera_factory(photom_assembly, config)

    screen = app.primaryScreen().availableGeometry()
    ctrl_window = PhotomApp(  # noqa: F841 -- keep a reference for the event loop
        photom_assembly=photom_assembly,
        photom_sensor_size_yx=config.camera_sensor_size_yx,
        photom_window_size_x=screen.width() // 3,
        photom_window_pos=(100, 100),
        demo_mode=demo,
        arduino=arduinos,
        camera_factory=camera_factory,
        calibration=CalibrationSettings(
            mirror_roi=[list(p) for p in config.calibration.mirror_roi],
            grid_n_points=config.calibration.grid_n_points,
            save_dir=config.calibration.save_dir,
        ),
        calibration_files=calibration_files,
    )
    return app.exec()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the photom GUI.")
    parser.add_argument("config", nargs="?", type=Path, help="photom YAML config")
    parser.add_argument("--demo", action="store_true", help="use simulated devices")
    parser.add_argument(
        "--calibration",
        action="append",
        type=Path,
        default=[],
        help="calibration YAML for the next mirror, in mirror order (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.config is None and not args.demo:
        parser.error("a config is required unless --demo is given")

    logging.basicConfig(level=logging.INFO)
    config = PhotomConfig.from_yaml(args.config) if args.config else None
    return launch(config, demo=args.demo, calibration_files=args.calibration)


if __name__ == "__main__":
    sys.exit(main())
