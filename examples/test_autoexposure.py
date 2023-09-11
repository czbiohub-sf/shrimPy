# %%
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import yaml

from skimage import data
from waveorder import visual

from mantis.acquisition.AcquisitionSettings import (
    AutoexposureSettings,
    ChannelSettings,
    MicroscopeSettings,
    SliceSettings,
)

from mantis.acquisition.autoexposure import (
    mean_intensity_autoexposure,
    intensity_percentile_autoexposure,
    masked_mean_intensity_autoexposure,
)

def create_autoexposure_test_dataset(img_stack):
    """
    Adjust the exposure of the image stack to create overexposed and
    underexposed stacks

    Parameters
    ----------
    img_stack : (C,Z,Y,X) Image stack
    Returns
    -------
    exposure_stack: with additional dimension containing:
                    under, over and nominally exposed image stacks
    """
    print("Creating Autoexposure Datase")
    C, Z, Y, X = img_stack.shape
    dtype = img_stack.dtype
    exposure_stack = np.zeros((3,) + img_stack.shape).astype(dtype)
    dtype_min = np.iinfo(dtype).min
    dtype_max = np.iinfo(dtype).max

    for c_idx in range(C):
        # Get the 5th and 95th percentile values
        pmin, pmax = np.percentile(img_stack[c_idx], (1, 90))
        # Under-exposed
        exposure_stack[0, c_idx] = np.where(
            img_stack[c_idx] > pmin, dtype_min, img_stack[c_idx]
        ).astype(dtype)
        # Over-exposed
        exposure_stack[1, c_idx] = np.where(
            img_stack[c_idx] > pmax, dtype_max, img_stack[c_idx]
        ).astype(dtype)
        # Nominaly-exposed
        exposure_stack[2, c_idx] = img_stack[c_idx].astype(dtype)
        # exposure_stack[2, c_idx] = np.where(
        #     (img_stack[c_idx] >= pmin) & (img_stack[c_idx] <= pmax),
        #     img_stack[c_idx],
        #     np.interp(img_stack[c_idx], [pmin, pmax], [dtype_min, dtype_max]).astype(
        #         dtype
        #     ),
        # )
    return exposure_stack


def plot_histograms(img_stack):
    """
    Plot the under,over and nominal exposed histograms for all channels

    Parameters
    ----------
    img_stack : Exposures,C,Z,Y,X
    """
    # create a subfigure with three columns
    fig, axes = plt.subplots(
        nrows=img_stack.shape[0], ncols=img_stack.shape[1], figsize=(12, 12)
    )
    for i in range(img_stack.shape[-5]):
        for j in range(img_stack.shape[-4]):
            # compute the histogram and bin values for the i-th image and j-th channel
            hist, bins = np.histogram(img_stack[i, j], bins=50)
            # select the axis for the i-th row and j-th column
            ax = axes[i, j]
            # plot the histogram
            ax.hist(img_stack[i, j].flatten(), bins=bins, alpha=0.5)
            ax.set_title(f"Image {i}, Channel {j}")
            ax.set_xlabel("Pixel Value")
            ax.set_ylabel("Frequency")
    # adjust the layout of the subfigure
    plt.tight_layout()


# %%
cells = data.cells3d().transpose((1, 0, 2, 3))
C, Z, Y, X = cells.shape
img_stack = create_autoexposure_test_dataset(cells)
# plot_histograms(img_stack)
# visual.image_stack_viewer(img_stack[:,:,30])

SETTINGS_PATH = (
    pathlib.Path(__file__).parent.parent
    / "acquisition"
    / "settings"
    / "demo_acquisition_settings.yaml"
)
with open(SETTINGS_PATH, "r") as file:
    mantis_settings = yaml.safe_load(file)
    print(mantis_settings)

channel_settings = ChannelSettings(**mantis_settings.get("ls_channel_settings"))
stack_settings = SliceSettings(**mantis_settings.get("ls_slice_settings"))
microscope_settings = MicroscopeSettings(**mantis_settings.get("ls_slice_settings"))
autoexposure_settings = AutoexposureSettings(
    **mantis_settings.get("autoexposure_settings")
)

# %%
methods = [mean_intensity_autoexposure, intensity_percentile_autoexposure, masked_mean_intensity_autoexposure]
for autoexposure_method in methods:
    # print(f"Using method: {method}")
    # Underexposure
    autoexposure_succeeded, new_laser_power, new_exposure = autoexposure_method(
        img_stack[0, 1],
        channel_settings.default_exposure_times_ms[0],
        channel_settings.default_laser_powers[0],
        autoexposure_settings,
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)

    autoexposure_succeeded, new_laser_power, new_exposure = autoexposure_method(
        img_stack[1, 1],
        channel_settings.default_exposure_times_ms[0],
        channel_settings.default_laser_powers[0],
        autoexposure_settings,
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)
    # assert autoexposure_succeeded is False
    # assert new_exposure > channel_settings.exposure_time_ms[0]
    autoexposure_succeeded, new_laser_power, new_exposure = autoexposure_method(
        img_stack[2, 1],
        channel_settings.default_exposure_times_ms[0],
        channel_settings.default_laser_powers[0],
        autoexposure_settings,
    )
    print(autoexposure_succeeded, new_laser_power, new_exposure)

# %%
plot_histograms(img_stack)
visual.image_stack_viewer(img_stack[:, :, 30])
