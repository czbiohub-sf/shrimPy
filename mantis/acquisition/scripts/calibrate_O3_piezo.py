### Methods
# Prepare a sample with dense beads immobilized dry on glass coverslip
# Image with epi illumination and LS detection
# As O3 moves back and forth the in-focus plate will shift up and down on the image

# %%
import numpy as np
from pycromanager import Core
from mantis.acquisition.microscope_operations import (
    acquire_defocus_stack,
    setup_kim101_stage,
)
import matplotlib.pyplot as plt

LS_KIM101_SN = 74000291

def smooth(data, window_size=5):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    data = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    return data

# %% Calibrate  background correction distance
# Acquire a number of defocus stacks and check that the stack starting and ending
# points don't drift up or down. Adjust the backlash correction distance if needed.
# This step is more important than adjusting the compensation factor below.

mmc = Core()
o3_stage = setup_kim101_stage(LS_KIM101_SN)

repeats = 10
z_range = np.arange(-165, 165, 15)
data = []
for _ in range(repeats):
    data.append(
        acquire_defocus_stack(
            mmc,
            z_stage=o3_stage,
            z_range=z_range,
            backlash_correction_distance=10,
        )
    )
data = np.concatenate(data)
o3_stage.close()

data_std = data.std(axis=-1)
peaks = []
for d in data_std:
    peaks.append(np.argmax(smooth(d, window_size=5)))

plt.plot(peaks)
plt.grid()

# %% Calibrate compensation factor
# Acquire a number of defocus stacks and check that the slope of the lines going
# in the forward and reverse directions is the same. Adjust the KIM101_COMPENSATION_FACTOR
# in mantis/acquisition/microscope_operations.py if needed

data_std = data.std(axis=-1)
peaks = []
for d in data_std:
    peaks.append(np.argmax(smooth(d, window_size=5)))

z_range = z_range.reshape(4, -1)
peaks = np.array(peaks).reshape(4, -1)
plt.plot(
     z_range.T,
     peaks.T
)

slope, intercept = [], []
for x, y in zip(z_range, peaks):
    p = np.polyfit(x, y, 1)
    slope.append(p[0])
    intercept.append(p[1])

print(f'Positive direction slope: {np.mean(slope[0::2])}')
print(f'Negative direction slope: {np.mean(slope[1::2])}')


# %%
