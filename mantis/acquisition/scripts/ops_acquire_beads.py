from pathlib import Path
from pycromanager import Acquisition, Core, multi_d_acquisition_events

mmc = Core()

acquisition_directory = Path(r'G:\OPS')
acquisition_name = 'OPS0078_beads'

z_start = -2
z_end = 25
z_step = 1

channel_name = '5-MultiCam_GFP_mCherry_BF'

mmc.set_property('Core', 'Focus', 'FocusDrive')
mmc.set_property('Prime', 'Exposure', '2.5')
mmc.set_property('Prime', 'Gain', '1-HDR')
mmc.set_property('BSI_Express', 'Exposure', '100')
mmc.set_property('BSI_Express', 'Gain', '1-HDR')
mmc.set_property('Zyla', 'Exposure', '100')
mmc.set_property('Andor ILE-A', 'Laser 561-Power Setpoint', '0')
mmc.set_property('Andor ILE-A', 'Laser 488-Power Setpoint', '0.1')

z0 = mmc.get_position('FocusDrive')

with Acquisition(directory=acquisition_directory, name=acquisition_name) as acq:
    acq.acquire(
        multi_d_acquisition_events(
            z_start=z_start+z0,
            z_end=z_end+z0,
            z_step=z_step,
            channel_group='Channels',
            channels=[channel_name],
            keep_shutter_open_between_z_steps=True,
        )
    )

mmc.set_property('Prime', 'Gain', '2-CMS')
mmc.set_property('BSI_Express', 'Gain', '2-CMS')
mmc.set_shutter_open(False)