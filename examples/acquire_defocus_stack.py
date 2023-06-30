from pycromanager import Core, Studio
from mantis.acquisition.microscope_operations import acquire_ls_defocus_stack

mmc = Core()
mmStudio = Studio()
z_stage = 'Z'
z_start = 0
z_end = 10
z_step = 1

data = acquire_ls_defocus_stack(mmc, mmStudio, z_stage, z_start, z_end, z_step)

print(data.shape)
