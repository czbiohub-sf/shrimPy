"""Launch the Mantis Acquisition GUI.

This script launches the MantisAcquisitionWidget, which provides a graphical
interface for configuring and running mantis microscope acquisitions.
"""

from qtpy.QtWidgets import QApplication

from shrimpy.mantis.mantis_acquisition_widget import MantisAcquisitionWidget
from shrimpy.mantis.mantis_v2 import initialize_mantis_core

if __name__ == "__main__":
    # Create Qt application
    app = QApplication([])

    # Initialize core using common function
    # Uncomment and modify path to load your microscope configuration
    # config_path = r"C:\Program Files\Micro-Manager-2.0\MMConfig_demo.cfg"
    # core = initialize_mantis_core(config_path)

    # Try to load mantis demo config
    try:
        demo_config = r"C:\Users\Cameron\justin\shrimPy\CompMicro_MMConfigs\Dev_Computer\mantis2-demo.cfg"
        core = initialize_mantis_core(demo_config)
        print(f"Loaded configuration: {demo_config}")
    except Exception as e:
        print(f"Could not load demo config: {e}")
        print("You may need to load a configuration manually from the GUI")
        core = None

    # Create and show the widget with the core instance
    widget = MantisAcquisitionWidget(core=core)
    widget.setWindowTitle("Mantis Acquisition Control")
    widget.resize(900, 700)
    widget.show()

    # Run the application
    app.exec()
