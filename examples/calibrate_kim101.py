# Calibration procedure 
# Image a 1 um fluorescent bead with epi illumination and LS detection. Focus O3
# on the bead. This script will defocus on one side of the bead and measure the
# image intensity. The stage calibration factor is determined from the
# difference in slope of average image intensity vs z position when traveling
# in the positive or negative direction 

#%%
import numpy as np
from pycromanager import Core, Studio
from mantis.acquisition.microscope_operations import setup_kim101_stage, acquire_ls_defocus_stack_and_display

#%%
mmc = Core()
mmStudio = Studio()
z_start = 0
z_end = 150
z_step = 15
galvo = 'AP Galvo'
galvo_range = [0]*5

z_stage = setup_kim101_stage('74000291')
z_range = np.hstack(
    (
        np.arange(z_start, z_end + z_step, z_step),
        np.arange(z_end, z_start - z_step, -z_step)
    )
)

#%%
data = acquire_ls_defocus_stack_and_display(
    mmc, 
    mmStudio, 
    z_stage, 
    z_range,
    galvo,
    galvo_range,
)

# %%
steps_per_direction = len(z_range)//2
intensity = data.sum(axis=(-1, -2))

pos_int = intensity[:, :steps_per_direction]
pos_z = z_range[:steps_per_direction]

neg_int = intensity[:, steps_per_direction:]
neg_z = z_range[steps_per_direction:]

A = np.vstack([pos_z, np.ones(len(pos_z))]).T
pos_slope = []
neg_slope = []
for i in range(len(galvo_range)):
    m, c = np.linalg.lstsq(A, pos_int[i], rcond=None)[0]
    pos_slope.append(m)
    m, c = np.linalg.lstsq(np.flipud(A), neg_int[i], rcond=None)[0]
    neg_slope.append(m)

compensation_factor = np.mean(pos_slope) / np.mean(neg_slope)
print(compensation_factor)

# %%
