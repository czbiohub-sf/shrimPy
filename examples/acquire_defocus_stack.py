from pycromanager import Core, Studio
from mantis.acquisition.microscope_operations import setup_kim101_stage, acquire_ls_defocus_stack

mmc = Core()
mmStudio = Studio()
z_start = 10
z_end = 100
z_step = 10
config_group = 'Channel - LS'
config_name = 'GFP EX488 EM525-45'

z_stage = setup_kim101_stage('74000291')

data = acquire_ls_defocus_stack(
    mmc, 
    mmStudio, 
    z_stage, 
    z_start, 
    z_end, 
    z_step, 
    config_group, 
    config_name,
)

print(data.shape)
