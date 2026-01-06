"""MantisAcquisitionWidget - Extended MDA widget with Mantis-specific settings.

This widget extends the standard MDA widget to include mantis-specific configuration
such as ROI settings, TriggerScope configuration, and other hardware parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pymmcore_plus.experimental.unicore import UniMMCore
from pymmcore_widgets import (
    MDAWidget,
    ImagePreview,
    StageWidget,
    LiveButton,
    SnapButton,
    CameraRoiWidget,
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

from mantis.acquisition.mantis_v2 import MantisEngine, initialize_mantis_core, create_mantis_engine


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
            'dac_sequencing': self.dac_device.text() if self.enable_dac_seq.isChecked() else None,
            'ttl_blanking': self.ttl_device.text() if self.enable_ttl_blanking.isChecked() else None,
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

    def __init__(self, core: UniMMCore | None = None, parent: QWidget | None = None):
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

    def __init__(self, core: UniMMCore | None = None, parent: QWidget | None = None):
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
    def __init__(self, core: UniMMCore | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._mmc = core
        self._mantis_engine: MantisEngine | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("<h2>Mantis Acquisition</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Main horizontal layout with three columns: preview, stage control, and tabs
        main_content = QHBoxLayout()

        # Left column: Image preview and ROI
        left_column = QVBoxLayout()
        
        preview_group = QGroupBox("Image Preview")
        preview_layout = QVBoxLayout()
        
        self.image_preview = ImagePreview()
        preview_layout.addWidget(self.image_preview)
        
        # Snap/Live buttons
        preview_buttons = QHBoxLayout()
        self.snap_button = SnapButton()
        self.live_button = LiveButton()
        preview_buttons.addWidget(self.snap_button)
        preview_buttons.addWidget(self.live_button)
        preview_buttons.addStretch()
        
        preview_layout.addLayout(preview_buttons)
        preview_group.setLayout(preview_layout)
        left_column.addWidget(preview_group, stretch=3)
        
        # ROI settings
        roi_group = QGroupBox("ROI Settings")
        roi_layout = QVBoxLayout()
        self.roi_widget = CameraRoiWidget()
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
                    self.xy_stage_widget = StageWidget(device=xy_stages[0], position_label_below=True)
                    stage_layout.addWidget(self.xy_stage_widget)
                
                if z_stages:
                    z_label = QLabel("<b>Z Stage</b>")
                    stage_layout.addWidget(z_label)
                    self.z_stage_widget = StageWidget(device=z_stages[0], position_label_below=True)
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
        self.mda_widget = MDAWidget()
        self.main_tabs.addTab(self.mda_widget, "Acquisition Sequence")

        # Mantis-specific settings
        self.mantis_settings = MantisSettingsWidget(core=self._mmc)
        #self.mantis_settings = MantisSettingsWidget()
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

        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addStretch()
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

    def _run_acquisition(self):
        """Execute the acquisition with mantis engine."""
        try:
            # Ensure we have a core instance
            if self._mmc is None:
                raise RuntimeError("No core instance available. Please load a configuration first.")
            
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
            sequence.metadata = sequence.metadata or {}
            sequence.metadata['mantis'] = mantis_settings

            # Create and register mantis engine if not already done
            if self._mantis_engine is None:
                use_hw_seq = mantis_settings.get('use_hardware_sequencing', True)
                self._mantis_engine = create_mantis_engine(self._mmc, use_hw_seq)

            self.status_label.setText("Running acquisition...")
            self.status_label.setStyleSheet("QLabel { color: blue; }")

            # Run the acquisition
            self._mmc.mda.run(sequence)

            self.status_label.setText("Acquisition complete!")
            self.status_label.setStyleSheet("QLabel { color: green; }")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")
            self.status_label.setStyleSheet("QLabel { color: red; }")
            raise

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
    #Initialize core using common function
    demo_config = r"C:\Users\Cameron\justin\shrimPy\CompMicro_MMConfigs\Dev_Computer\mantis2-demo.cfg"
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
