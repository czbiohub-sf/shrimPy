"""MantisAcquisitionWidget - Extended MDA widget with Mantis-specific settings.

This widget extends the standard MDA widget to include mantis-specific configuration
such as ROI settings, TriggerScope configuration, and other hardware parameters.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from pymmcore_plus import CMMCorePlus
from pymmcore_widgets import (
    CameraRoiWidget,
    ImagePreview,
    LiveButton,
    MDAWidget,
    SnapButton,
    StageWidget,
)
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from useq import MDASequence

from shrimpy.mantis.mantis_engine import MantisEngine, create_mantis_engine, initialize_mantis_core


class CustomCameraRoiWidget(CameraRoiWidget):
    """CameraRoiWidget that doesn't use snap() - uses continuous acquisition instead.

    This subclass overrides the auto-snap functionality to work around PVCAM camera
    buffer issues when using snap() command. Instead of snap(), it uses continuous
    acquisition to capture a single frame.
    """

    def _on_roi_set(self, camera: str, x: int, y: int, width: int, height: int) -> None:
        """Handle the ROI set event without calling snap()."""
        if camera != self.camera:
            self._update_unselected_camera_info(camera, x, y, width, height)
            return

        # if the roi is not centered, uncheck the center checkbox
        from superqt.utils import signals_blocked

        centered = (
            x == (self._cameras[camera].pixel_width - width) // 2
            and y == (self._cameras[camera].pixel_height - height) // 2
        )
        with signals_blocked(self.center_checkbox):
            self.center_checkbox.setChecked(centered)

        # update the roi values in the spinboxes
        from pymmcore_widgets.control._camera_roi_widget import ROI

        self._update_roi_values(ROI(x, y, width, height, centered))

        # update the crop mode combo box text to match the set roi (this is mainly
        # needed when the roi is set from the core)
        crop_mode = self._get_updated_crop_mode(camera, *self._get_roi_values())
        with signals_blocked(self.camera_roi_combo):
            self.camera_roi_combo.setCurrentText(crop_mode)

        # update the stored camera info
        self._cameras[camera] = self._cameras[camera].replace(
            crop_mode=crop_mode, roi=ROI(x, y, width, height, centered)
        )

        self._custom_roi_wdg.setEnabled(crop_mode == "Custom ROI")
        self.crop_btn.setEnabled(crop_mode == "Custom ROI")

        self._update_lbl_info()

        # REMOVED: Auto-snap functionality (causes buffer issues with PVCAM cameras)
        # if self.snap_checkbox.isChecked() and self.snap_checkbox.isVisible():
        #     self._mmc.snap()

        self.roiChanged.emit(x, y, width, height, crop_mode)


class ROISettingsWidget(QWidget):
    """Widget for configuring camera ROI settings."""

    valueChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        # ROI parameters (x, y, width, height)
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 4096)
        self.x_spin.setValue(0)
        self.x_spin.valueChanged.connect(self.valueChanged.emit)

        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 4096)
        self.y_spin.setValue(512)
        self.y_spin.valueChanged.connect(self.valueChanged.emit)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 4096)
        self.width_spin.setValue(2048)
        self.width_spin.valueChanged.connect(self.valueChanged.emit)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 4096)
        self.height_spin.setValue(256)
        self.height_spin.valueChanged.connect(self.valueChanged.emit)

        layout.addRow("X Offset:", self.x_spin)
        layout.addRow("Y Offset:", self.y_spin)
        layout.addRow("Width:", self.width_spin)
        layout.addRow("Height:", self.height_spin)

        # Preset buttons
        preset_layout = QHBoxLayout()
        self.full_fov_btn = QPushButton("Full FOV")
        self.full_fov_btn.clicked.connect(self._set_full_fov)
        self.centered_btn = QPushButton("Centered Strip")
        self.centered_btn.clicked.connect(self._set_centered_strip)

        preset_layout.addWidget(self.full_fov_btn)
        preset_layout.addWidget(self.centered_btn)
        layout.addRow("Presets:", preset_layout)

    def _set_full_fov(self):
        """Set ROI to full field of view."""
        self.x_spin.setValue(0)
        self.y_spin.setValue(0)
        self.width_spin.setValue(2048)
        self.height_spin.setValue(2048)

    def _set_centered_strip(self):
        """Set ROI to centered strip (mantis default)."""
        self.x_spin.setValue(0)
        self.y_spin.setValue(512)
        self.width_spin.setValue(2048)
        self.height_spin.setValue(256)

    def value(self) -> list[int]:
        """Get ROI as [x, y, width, height]."""
        return [
            self.x_spin.value(),
            self.y_spin.value(),
            self.width_spin.value(),
            self.height_spin.value(),
        ]

    def setValue(self, roi: list[int]):
        """Set ROI from [x, y, width, height]."""
        if len(roi) == 4:
            self.x_spin.setValue(roi[0])
            self.y_spin.setValue(roi[1])
            self.width_spin.setValue(roi[2])
            self.height_spin.setValue(roi[3])


class TriggerScopeSettingsWidget(QWidget):
    """Widget for configuring TriggerScope hardware triggering."""

    valueChanged = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        # DAC sequencing device
        self.dac_device = QLineEdit("TS_DAC01")
        self.dac_device.textChanged.connect(self.valueChanged.emit)
        layout.addRow("DAC Sequencing Device:", self.dac_device)

        # TTL blanking device
        self.ttl_device = QLineEdit("TS_TTL1-8")
        self.ttl_device.textChanged.connect(self.valueChanged.emit)
        layout.addRow("TTL Blanking Device:", self.ttl_device)

        # Enable/disable checkboxes
        self.enable_dac_seq = QCheckBox("Enable DAC Sequencing")
        self.enable_dac_seq.setChecked(True)
        self.enable_dac_seq.stateChanged.connect(self.valueChanged.emit)
        layout.addRow(self.enable_dac_seq)

        self.enable_ttl_blanking = QCheckBox("Enable TTL Blanking")
        self.enable_ttl_blanking.setChecked(True)
        self.enable_ttl_blanking.stateChanged.connect(self.valueChanged.emit)
        layout.addRow(self.enable_ttl_blanking)

    def value(self) -> dict[str, Any]:
        """Get TriggerScope settings as dictionary."""
        return {
            'dac_sequencing': self.dac_device.text()
            if self.enable_dac_seq.isChecked()
            else None,
            'ttl_blanking': self.ttl_device.text()
            if self.enable_ttl_blanking.isChecked()
            else None,
        }

    def setValue(self, settings: dict[str, Any]):
        """Set TriggerScope settings from dictionary."""
        if dac := settings.get('dac_sequencing'):
            self.dac_device.setText(dac)
            self.enable_dac_seq.setChecked(True)
        else:
            self.enable_dac_seq.setChecked(False)

        if ttl := settings.get('ttl_blanking'):
            self.ttl_device.setText(ttl)
            self.enable_ttl_blanking.setChecked(True)
        else:
            self.enable_ttl_blanking.setChecked(False)


class MicroscopeSettingsWidget(QWidget):
    """Widget for additional microscope configuration."""

    valueChanged = Signal()

    def __init__(self, core: CMMCorePlus | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._mmc = core
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)

        # Focus device selection
        self.focus_device = QComboBox()
        self._populate_focus_devices()
        self.focus_device.currentTextChanged.connect(self.valueChanged.emit)
        layout.addRow("Focus Device:", self.focus_device)

        # Autofocus settings
        self.use_autofocus = QCheckBox("Enable Autofocus")
        self.use_autofocus.stateChanged.connect(self.valueChanged.emit)
        layout.addRow(self.use_autofocus)

        self.autofocus_method = QComboBox()
        self.autofocus_method.addItems(["PFS", "Hardware", "Software"])
        self.autofocus_method.currentTextChanged.connect(self.valueChanged.emit)
        layout.addRow("Autofocus Method:", self.autofocus_method)

        # Hardware sequencing preferences
        self.use_hw_seq = QCheckBox("Use Hardware Sequencing")
        self.use_hw_seq.setChecked(True)
        self.use_hw_seq.setToolTip(
            "Enable hardware-triggered sequences for faster acquisition"
        )
        self.use_hw_seq.stateChanged.connect(self.valueChanged.emit)
        layout.addRow(self.use_hw_seq)

    def _populate_focus_devices(self):
        """Populate focus device dropdown from loaded config."""
        try:
            if self._mmc is not None:
                devices = [self._mmc.getFocusDevice()]
                # Try to get all stage devices
                for dev in self._mmc.getLoadedDevices():
                    dev_type = self._mmc.getDeviceType(dev)
                    if dev_type == 1:  # Stage device type
                        if dev not in devices:
                            devices.append(dev)
                self.focus_device.addItems(devices)
            else:
                self.focus_device.addItems(["AP Galvo", "ZDrive", "Z"])
        except Exception:
            self.focus_device.addItems(["AP Galvo", "ZDrive", "Z"])

    def value(self) -> dict[str, Any]:
        """Get microscope settings as dictionary."""
        return {
            'focus_device': self.focus_device.currentText(),
            'autofocus': {
                'enabled': self.use_autofocus.isChecked(),
                'method': self.autofocus_method.currentText(),
            },
            'use_hardware_sequencing': self.use_hw_seq.isChecked(),
        }

    def setValue(self, settings: dict[str, Any]):
        """Set microscope settings from dictionary."""
        if focus_dev := settings.get('focus_device'):
            idx = self.focus_device.findText(focus_dev)
            if idx >= 0:
                self.focus_device.setCurrentIndex(idx)

        if af_settings := settings.get('autofocus'):
            self.use_autofocus.setChecked(af_settings.get('enabled', False))
            method = af_settings.get('method', 'PFS')
            idx = self.autofocus_method.findText(method)
            if idx >= 0:
                self.autofocus_method.setCurrentIndex(idx)

        self.use_hw_seq.setChecked(settings.get('use_hardware_sequencing', True))


class MantisSettingsWidget(QWidget):
    """Composite widget for all Mantis-specific settings."""

    valueChanged = Signal()

    def __init__(self, core: CMMCorePlus | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._core = core
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Create tab widget for organized settings
        self.tabs = QTabWidget()

        # TriggerScope settings tab
        self.triggerscope_widget = TriggerScopeSettingsWidget()
        self.triggerscope_widget.valueChanged.connect(self.valueChanged.emit)
        self.tabs.addTab(self.triggerscope_widget, "TriggerScope")

        # Microscope settings tab
        self.microscope_widget = MicroscopeSettingsWidget(core=self._core)
        self.microscope_widget.valueChanged.connect(self.valueChanged.emit)
        self.tabs.addTab(self.microscope_widget, "Microscope")

        layout.addWidget(self.tabs)

    def value(self) -> dict[str, Any]:
        """Get all mantis settings as a dictionary."""
        return {
            'trigger_scope': self.triggerscope_widget.value(),
            **self.microscope_widget.value(),
        }

    def setValue(self, settings: dict[str, Any]):
        """Set all mantis settings from a dictionary."""
        if ts := settings.get('trigger_scope'):
            self.triggerscope_widget.setValue(ts)
        self.microscope_widget.setValue(settings)


class MantisAcquisitionWidget(QWidget):
    """Complete acquisition widget with standard MDA + Mantis settings.

    This widget combines the standard pymmcore-widgets MDAWidget with
    mantis-specific configuration options.
    """

    def __init__(self, core: CMMCorePlus | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        # If no core provided, get or create the singleton instance
        self._mmc = core if core is not None else CMMCorePlus.instance()
        self._mantis_engine: MantisEngine | None = None
        self._is_paused = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Mantis Acquisition</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Check if camera is available and warn if not
        camera_available = False
        camera_warning = None
        if self._mmc is not None:
            try:
                camera = self._mmc.getCameraDevice()
                camera_available = bool(camera)
                # Test if camera can actually snap
                if camera_available:
                    try:
                        # Quick test - try to get exposure (doesn't trigger camera)
                        _ = self._mmc.getExposure()
                    except Exception as e:
                        camera_warning = (
                            f"Camera '{camera}' detected but may not be functional: {str(e)}"
                        )
            except Exception as e:
                camera_warning = f"Camera detection failed: {str(e)}"

        if not camera_available or camera_warning:
            msg = (
                camera_warning
                if camera_warning
                else "⚠️ Warning: No camera device detected. Image acquisition will not work."
            )
            warning_label = QLabel(msg)
            warning_label.setStyleSheet(
                "QLabel { background-color: #FFF3CD; color: #856404; "
                "padding: 8px; border: 1px solid #FFC107; border-radius: 4px; }"
            )
            warning_label.setWordWrap(True)
            layout.addWidget(warning_label)

        # Main horizontal layout with three columns: preview, stage control, and tabs
        main_content = QHBoxLayout()

        # Left column: Image preview and ROI
        left_column = QVBoxLayout()

        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()

        self.image_preview = ImagePreview(mmcore=self._mmc)
        preview_layout.addWidget(self.image_preview)

        # Snap/Live buttons
        preview_buttons = QHBoxLayout()
        self.snap_button = SnapButton(mmcore=self._mmc)
        # Override snap button behavior for PVCAM cameras that don't support standard snap()
        self.snap_button.clicked.disconnect()
        self.snap_button.clicked.connect(self._custom_snap)
        self.live_button = LiveButton(mmcore=self._mmc)
        preview_buttons.addWidget(self.snap_button)
        preview_buttons.addWidget(self.live_button)
        preview_buttons.addStretch()

        preview_layout.addLayout(preview_buttons)
        preview_group.setLayout(preview_layout)
        left_column.addWidget(preview_group, stretch=3)

        # ROI settings
        roi_group = QGroupBox("ROI Settings")
        roi_layout = QVBoxLayout()
        self.roi_widget = CustomCameraRoiWidget(mmcore=self._mmc)
        roi_layout.addWidget(self.roi_widget)
        roi_group.setLayout(roi_layout)
        left_column.addWidget(roi_group, stretch=1)

        main_content.addLayout(left_column, stretch=1)

        # Middle column: Stage control
        stage_group = QGroupBox("Stage Control")
        stage_layout = QVBoxLayout()

        # XY Stage control
        try:
            from pymmcore_plus import DeviceType

            if self._mmc is not None:
                xy_stages = list(self._mmc.getLoadedDevicesOfType(DeviceType.XYStage))
                z_stages = list(self._mmc.getLoadedDevicesOfType(DeviceType.Stage))

                if xy_stages:
                    xy_label = QLabel("<b>XY Stage</b>")
                    stage_layout.addWidget(xy_label)
                    self.xy_stage_widget = StageWidget(
                        device=xy_stages[0], position_label_below=True, mmcore=self._mmc
                    )
                    stage_layout.addWidget(self.xy_stage_widget)

                if z_stages:
                    z_label = QLabel("<b>Z Stage</b>")
                    stage_layout.addWidget(z_label)
                    self.z_stage_widget = StageWidget(
                        device=z_stages[0], position_label_below=True, mmcore=self._mmc
                    )
                    stage_layout.addWidget(self.z_stage_widget)

                if not xy_stages and not z_stages:
                    no_stage_label = QLabel("No stages detected")
                    no_stage_label.setStyleSheet("color: gray; font-style: italic;")
                    stage_layout.addWidget(no_stage_label)
            else:
                no_core_label = QLabel("No core instance loaded")
                no_core_label.setStyleSheet("color: gray; font-style: italic;")
                stage_layout.addWidget(no_core_label)
        except Exception as e:
            error_label = QLabel(f"Stage initialization error: {str(e)}")
            error_label.setStyleSheet("color: red; font-size: 10px;")
            stage_layout.addWidget(error_label)

        stage_layout.addStretch()
        stage_group.setLayout(stage_layout)
        main_content.addWidget(stage_group, stretch=1)

        # Right column: Main content in tabs
        self.main_tabs = QTabWidget()

        # Standard MDA widget
        self.mda_widget = MDAWidget(mmcore=self._mmc)
        self.main_tabs.addTab(self.mda_widget, "Acquisition Sequence")

        # Mantis-specific settings
        self.mantis_settings = MantisSettingsWidget(core=self._mmc)
        # self.mantis_settings = MantisSettingsWidget()
        self.main_tabs.addTab(self.mantis_settings, "Mantis Settings")

        main_content.addWidget(self.main_tabs, stretch=2)

        layout.addLayout(main_content)

        # Control buttons
        button_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Settings...")
        self.save_btn = QPushButton("Save Settings...")
        self.run_btn = QPushButton("▶ Run Acquisition")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn = QPushButton("⏸ Pause")
        self.pause_btn.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px; }"
        )
        self.pause_btn.setEnabled(False)  # Disabled until acquisition starts

        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.pause_btn)
        button_layout.addWidget(self.run_btn)

        layout.addLayout(button_layout)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        layout.addWidget(self.status_label)

    def _connect_signals(self):
        """Connect widget signals."""
        self.load_btn.clicked.connect(self._load_settings)
        self.save_btn.clicked.connect(self._save_settings)
        self.run_btn.clicked.connect(self._run_acquisition)
        self.pause_btn.clicked.connect(self._toggle_pause)

        # Connect to MDA signals to update pause button state
        if self._mmc is not None:
            self._mmc.mda.events.sequenceStarted.connect(self._on_acquisition_started)
            self._mmc.mda.events.sequenceFinished.connect(self._on_acquisition_finished)
            self._mmc.mda.events.sequencePauseToggled.connect(self._on_pause_toggled)

    def _custom_snap(self):
        """Custom snap implementation using continuous acquisition.

        This is needed for PVCAM cameras (like Prime BSI Express) that have issues
        with the standard snap() command due to buffer management problems in the
        PVCAM adapter. The camera works perfectly in continuous mode, so we use
        that to acquire a single frame.

        Root cause: The PVCAM adapter's SnapImage() completes but GetImageBuffer()
        fails to return valid data from singleFrameBufFinal_, likely due to buffer
        initialization issues when CircularBufferEnabled is ON.
        """
        if self._mmc is None:
            return

        try:
            import time

            # Stop any running sequence
            if self._mmc.isSequenceRunning():
                self._mmc.stopSequenceAcquisition()
                time.sleep(0.05)

            # Use continuous acquisition to get a single frame
            self._mmc.initializeCircularBuffer()
            self._mmc.startContinuousSequenceAcquisition(0)
            time.sleep(0.1)  # Wait for at least one frame

            # Check if we got an image
            if self._mmc.getRemainingImageCount() > 0:
                # Get the image from the buffer before stopping
                img = self._mmc.getLastImage()

                # Stop the sequence now that we have the image
                self._mmc.stopSequenceAcquisition()

                # Directly update the ImagePreview instead of emitting signal
                # This avoids the signal handler trying to call getImage() which expects snap mode
                self.image_preview._update_image(img)
            else:
                self._mmc.stopSequenceAcquisition()
                print("No images in buffer")

        except Exception as e:
            print(f"Snap failed: {e}")
            if self._mmc.isSequenceRunning():
                self._mmc.stopSequenceAcquisition()

    def _run_acquisition(self):
        """Execute the acquisition with mantis engine."""
        try:
            # Ensure we have a core instance
            if self._mmc is None:
                raise RuntimeError(
                    "No core instance available. Please load a configuration first."
                )

            # Get the MDA sequence from the widget
            sequence = self.mda_widget.value()

            # Add mantis metadata to the sequence
            mantis_settings = self.mantis_settings.value()
            # CameraRoiWidget manages ROI directly with the camera, but we can still save it
            # Get ROI from camera if available
            if self._mmc is not None:
                try:
                    x = self._mmc.getROI()[0]
                    y = self._mmc.getROI()[1]
                    w = self._mmc.getROI()[2]
                    h = self._mmc.getROI()[3]
                    mantis_settings['roi'] = [x, y, w, h]
                except Exception:
                    pass

            # Create new sequence with updated metadata (MDASequence is frozen)
            updated_metadata = dict(sequence.metadata or {})
            updated_metadata['mantis'] = mantis_settings
            sequence = sequence.model_copy(update={'metadata': updated_metadata})

            # Create and register mantis engine if not already done
            if self._mantis_engine is None:
                use_hw_seq = mantis_settings.get('use_hardware_sequencing', True)
                self._mantis_engine = create_mantis_engine(self._mmc, use_hw_seq)

            self.status_label.setText("Running acquisition...")
            self.status_label.setStyleSheet("QLabel { color: blue; }")

            # Run the acquisition in a separate thread to keep GUI responsive
            # The acquisition complete status will be updated by the sequenceFinished signal
            self._mmc.run_mda(sequence, block=False)

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            raise

    def _toggle_pause(self):
        """Toggle pause/resume for the current acquisition."""
        if self._mmc is not None:
            self._mmc.mda.toggle_pause()

    def _on_acquisition_started(self):
        """Called when acquisition starts."""
        self.pause_btn.setEnabled(True)
        self._is_paused = False
        self.pause_btn.setText("⏸ Pause")

    def _on_acquisition_finished(self):
        """Called when acquisition finishes."""
        self.pause_btn.setEnabled(False)
        self._is_paused = False
        self.pause_btn.setText("⏸ Pause")
        self.status_label.setText("Acquisition complete!")
        self.status_label.setStyleSheet("QLabel { color: green; }")

    def _on_pause_toggled(self, paused: bool):
        """Called when acquisition is paused or resumed."""
        self._is_paused = paused
        if paused:
            self.pause_btn.setText("▶ Resume")
            self.status_label.setText("Acquisition paused")
            self.status_label.setStyleSheet("QLabel { color: orange; }")
        else:
            self.pause_btn.setText("⏸ Pause")
            self.status_label.setText("Running acquisition...")
            self.status_label.setStyleSheet("QLabel { color: blue; }")

    def _save_settings(self):
        """Save current settings to a YAML file."""
        from qtpy.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Acquisition Settings",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )

        if filename:
            try:
                # Get MDA sequence and add mantis metadata
                sequence = self.mda_widget.value()
                mantis_settings = self.mantis_settings.value()
                # Get current ROI from camera
                if self._mmc is not None:
                    try:
                        x = self._mmc.getROI()[0]
                        y = self._mmc.getROI()[1]
                        w = self._mmc.getROI()[2]
                        h = self._mmc.getROI()[3]
                        mantis_settings['roi'] = [x, y, w, h]
                    except Exception:
                        pass

                # Create combined settings dictionary
                sequence_dict = sequence.dict()
                sequence_dict['metadata'] = sequence_dict.get('metadata', {})
                sequence_dict['metadata']['mantis'] = mantis_settings

                # Convert tuples to lists to avoid !!python/tuple tags
                sequence_dict = self._sanitize_for_yaml(sequence_dict)

                # Save to YAML using safe_dump to avoid Python-specific tags
                with open(filename, 'w') as f:
                    yaml.safe_dump(
                        sequence_dict,
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                    )

                self.status_label.setText(f"Settings saved to {Path(filename).name}")
                self.status_label.setStyleSheet("QLabel { color: green; }")

            except Exception as e:
                self.status_label.setText(f"Error saving: {str(e)}")
                self.status_label.setStyleSheet("QLabel { color: red; }")

    def _sanitize_for_yaml(self, obj: Any) -> Any:
        """Convert Python-specific types to basic types for clean YAML output.

        This converts tuples to lists and handles nested structures recursively.
        """
        if isinstance(obj, tuple):
            return [self._sanitize_for_yaml(item) for item in obj]
        elif isinstance(obj, list):
            return [self._sanitize_for_yaml(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_yaml(value) for key, value in obj.items()}
        else:
            return obj

    def _load_settings(self):
        """Load settings from a YAML file."""
        from qtpy.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Acquisition Settings",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )

        if filename:
            try:
                # Load the sequence from file
                sequence = MDASequence.from_file(filename)

                # Set the MDA widget value
                self.mda_widget.setValue(sequence)

                # Extract and set mantis settings from metadata
                if sequence.metadata and 'mantis' in sequence.metadata:
                    mantis_meta = sequence.metadata['mantis']
                    self.mantis_settings.setValue(mantis_meta)
                    # Apply ROI to camera if available
                    if 'roi' in mantis_meta and self._mmc is not None:
                        try:
                            roi = mantis_meta['roi']
                            if len(roi) == 4:
                                self._mmc.setROI(*roi)
                        except Exception as e:
                            print(f"Could not set ROI: {e}")

                self.status_label.setText(f"Settings loaded from {Path(filename).name}")
                self.status_label.setStyleSheet("QLabel { color: green; }")

            except Exception as e:
                self.status_label.setText(f"Error loading: {str(e)}")
                self.status_label.setStyleSheet("QLabel { color: red; }")


# Example standalone application
if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    # Initialize core using common function
    demo_config = (
        r"C:\Users\Cameron\justin\shrimPy\CompMicro_MMConfigs\Dev_Computer\mantis2-demo.cfg"
    )
    try:
        core = initialize_mantis_core(demo_config)
        print(f"Loaded configuration: {demo_config}")
    except Exception as e:
        print(f"Could not load config: {e}")
        print("Continuing without config...")
        core = None

    # Create and show widget with the core instance
    widget = MantisAcquisitionWidget(core=core)
    widget.setWindowTitle("Mantis Acquisition Control")
    widget.resize(1400, 800)  # Larger size to accommodate image preview and stage control
    widget.show()

    app.exec()
