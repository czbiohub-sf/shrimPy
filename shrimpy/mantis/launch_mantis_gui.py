"""Launch the Mantis Acquisition GUI.

This script launches the MantisAcquisitionWidget, which provides a graphical
interface for configuring and running mantis microscope acquisitions.
"""

from pymmcore_plus import CMMCorePlus
from qtpy.QtWidgets import QApplication

from shrimpy.mantis.mantis_acquisition_widget import MantisAcquisitionWidget

if __name__ == "__main__":
    # Create Qt application
    app = QApplication([])

    # Try to load mantis demo config
    try:
        demo_config = r"C:\Users\Cameron\justin\shrimPy\CompMicro_MMConfigs\Dev_Computer\mantis2-demo.cfg"
        core = CMMCorePlus()
        core.loadSystemConfiguration(demo_config)
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
