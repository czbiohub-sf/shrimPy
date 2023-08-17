import os
import warnings

from pathlib import Path

import click
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from iohub.display_utils import channel_display_settings
from iohub.ngff import open_ome_zarr
from scipy.optimize import curve_fit
from tqdm import tqdm

from mantis.cli.parsing import input_position_dirpaths, output_dirpath

MSECS_PER_MINUTE = 60000


def plot_bleaching_curves(times, tczyx_data, channel_names, output_file, title=''):
    """Plots bleaching curves and estimates bleaching lifetimes

    Parameters
    ----------
    tc_times : NDArray with shape (T,)
        Times of acquisition for each time point (minutes)
    tczyx_data : NDArray with shape (T, C, Z, Y, X)
        Raw data
    channel_names : list of strings with length (C)
    output_file : str
    title : str, optional
        plot title, by default ''
    """
    num_times = tczyx_data.shape[0]
    num_channels = tczyx_data.shape[1]

    means = np.zeros((num_times, num_channels))
    stds = np.zeros_like(means)

    # Calculate statistics
    for t in tqdm(range(num_times)):
        for channel_index in range(num_channels):
            zyx_data = tczyx_data[t, channel_index, ...]  # zyx
            means[t, channel_index] = np.mean(zyx_data)
            stds[t, channel_index] = np.std(zyx_data)

    # Generate and save plots
    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    for channel_index in range(num_channels):
        channel_color = matplotlib.colors.to_rgb(
            "#" + channel_display_settings(channel_names[channel_index]).color
        )

        xdata = times[:]
        ydata = means[:, channel_index]
        yerr = stds[:, channel_index]

        # Plot curve fit
        def func(x, a, b, c):
            return a * np.exp(-x / b) + c

        try:
            popt, _ = curve_fit(
                func,
                xdata,
                ydata,
                sigma=yerr,
                p0=(np.max(ydata) - np.min(ydata), 100, np.min(ydata)),
                maxfev=5000,
            )

            xx = np.linspace(0, np.max(xdata), 100)
            ax.plot(xx, func(xx, *popt), color=channel_color, alpha=0.5)
            label = channel_names[channel_index] + f" - {popt[1]:0.0f} minutes"
            print("Curve fit successful!")
            print(label)
        except Exception as e:
            print(e)
            label = channel_names[channel_index]
            print("Curve fit failed!")

        # Plot data
        ax.plot(
            xdata,
            ydata,
            label=label,
            marker='o',
            markeredgewidth=0,
            linewidth=0,
            color=channel_color,
        )

    ax.set_title(title, {'fontsize': 8})
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Mean Intensity (AU)")
    ax.legend(frameon=False, markerfirst=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


@click.command()
@input_position_dirpaths()
@output_dirpath()
def estimate_bleaching(input_position_dirpaths, output_dirpath):
    """
    Estimate bleaching from raw data

    >> mantis estimate-bleaching -i ./input.zarr/0/0/0 -o ./bleaching-curves/
    """

    # Read plate metadata if it exists
    try:
        plate_path = Path(*Path(input_position_dirpaths[0]).parts[:-3])
        with open_ome_zarr(plate_path) as plate_reader:
            plate_zattrs = plate_reader.zattrs
    except Exception as e:
        print(e)
        warnings.warn(
            "WARNING: this position has no plate metadata, so the time metadata will be missing."
        )

    # Loop through position
    for input_position_dirpath in input_position_dirpaths:
        with open_ome_zarr(input_position_dirpath) as reader:
            well_name = "/".join(Path(input_position_dirpath).parts[-3:])
            tczyx_data = reader["0"]

        print(f"Generating bleaching curves for position {well_name}")

        # Generate plot for each position
        T = tczyx_data.shape[0]
        try:
            dt = np.float32(plate_zattrs['Summary']['Interval_ms'] / MSECS_PER_MINUTE)
        except Exception as e:
            print(e)
            warnings.warn(f"WARNING: missing time metadata for p={well_name}")
            dt = 1

        times = np.arange(0, T * dt, step=dt)
        output_file = os.path.join(output_dirpath, well_name)
        os.makedirs(output_file, exist_ok=True)
        title = str(input_position_dirpath) + f" - position = {well_name}"
        plot_bleaching_curves(
            times,
            tczyx_data,
            reader.channel_names,
            os.path.join(output_file, "bleaching.svg"),
            title,
        )

    reader.close()
