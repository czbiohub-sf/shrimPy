"""Photomanipulation: lasers steered by galvo mirrors and calibrated to a camera.

Only the hardware-facing assembly and its calibration helpers are vendored,
not coPylot's photom Qt GUI. Moving mirrors needs only the core dependencies;
camera calibration (``PhotomAssembly.calibrate_w_camera``) needs the
``calibration`` extra (scikit-image, tifffile, matplotlib).
"""
