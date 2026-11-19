# copylot (photom subset)

The parts of [coPylot](https://github.com/czbiohub-sf/coPylot) that the photom
(photomanipulation) setup uses, vendored from coPylot main @ `0bcc3a3`
(BSD-3-Clause) and maintained here as a uv workspace member of shrimPy. coPylot
itself is being archived, so **this is the maintained copy**, not a mirror of
upstream.

| Module | Contents |
| --- | --- |
| `copylot.hardware.lasers.abstract_laser` | `AbstractLaser` interface |
| `copylot.hardware.lasers.vortran.vortran` | `VortranLaser`: Vortran Stradus over RS-232 |
| `copylot.hardware.mirrors.abstract_mirror` | `AbstractMirror` interface |
| `copylot.hardware.mirrors.optotune.mirror` | `OptoMirror`: Optotune MR-E-2 steering mirror |
| `copylot.assemblies.photom.photom` | `PhotomAssembly`: galvo-steered lasers with camera calibration |
| `copylot.assemblies.photom.utils` | affine transform, spot detection, calibration grid, ROI patterns, Arduino PWM (+ sketches) |
| `copylot.assemblies.photom.photom_mock_devices` | simulated lasers, mirror, Arduino and calibration camera |
| `copylot.assemblies.photom.gui` | the photom Qt GUI (qtpy) |

coPylot's module paths are kept, so code written against it
(`from copylot.hardware.lasers.vortran.vortran import VortranLaser`) works
unchanged. Each module's docstring lists its fixes relative to coPylot. Because the
distribution is named `copylot`, it replaces coPylot in an environment.

## Extras

- `calibration`: scikit-image, tifffile, matplotlib, for `PhotomAssembly.calibrate_w_camera`
- `gui`: qtpy + PyQt6, for the photom GUI
- `optotune`: Optotune's MR-E Python SDK (`optoMDC`, `optoKummenberg`), for `OptoMirror`

The Optotune SDK is not on any package index. In a shrimPy checkout, uv installs
Optotune's unmodified wheels from `third_party/optotune/` (see its README for
provenance and Optotune's license). With pip, download the SDK from
<https://www.optotune.com/software-download/> and `pip install` its two wheels
first.

## Photom GUI

```bash
shrimpy gui photom --demo                       # simulated devices, from shrimPy
shrimpy gui photom --config photom_VIS.yaml     # real hardware
python -m copylot.assemblies.photom.gui --demo  # without shrimPy

# start with a saved camera-to-mirror calibration (one per mirror, in order)
shrimpy gui photom --config photom_VIS.yaml --calibration affine_VIS.yml
```

The control panel ("Calibration: ...") and the Laser Marker window title show the
current mirror's calibration file. A calibration can come from the mirror's
`affine_matrix_path` in the config, `--calibration` (which overrides it), the Load
Calibration button, or Calibrate; all of them are applied the same way.

The config (`PhotomConfig` in `copylot.assemblies.photom.gui.photom_gui`) keeps
coPylot's keys (`lasers`, `mirrors`, `COM_port`, `affine_matrix_path`, ...) and adds
`arduinos`, `camera_sensor_size_yx` and `calibration`; shrimPy's
`config/photom/photom_VIS.yaml` is an annotated example.

Calibration needs a camera. Demo mode simulates one. For real hardware, no camera
is wired up yet (coPylot's FLIR camera driver was not vendored), so the Calibrate
button is disabled and calibrations are loaded from file. `launch(camera_factory=...)`
accepts any object with `image_size` and `snap()`.

## Install on its own

```bash
pip install "copylot[calibration,gui] @ git+https://github.com/czbiohub-sf/shrimPy.git#subdirectory=packages/copylot"
```
