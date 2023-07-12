import numpy as np
from pycromanager import Core, Studio
from mantis.acquisition.microscope_operations import setup_kim101_stage
from mantis.acquisition.acq_engine import acquire_ls_defocus_stack

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(levelname)s - %(module)s.%(funcName)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

mmc = Core()
mmStudio = Studio()
z_start = -200
z_end = 200
z_step = 25
config_group = 'Channel - LS'
config_name = 'GFP EX488 EM525-45'
galvo = 'AP Galvo'
galvo_range = [-0.5, 0, 0.5]

z_stage = setup_kim101_stage('74000291')
z_range = np.arange(z_start, z_end + z_step, z_step)

data = acquire_ls_defocus_stack(
    mmc, 
    mmStudio, 
    z_stage, 
    z_range,
    galvo,
    galvo_range,
    config_group, 
    config_name,
    close_display=False,
)

print(data.shape)
