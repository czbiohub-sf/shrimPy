import os

import click
import matplotlib.pyplot as plt
import numpy as np

from iohub import read_micromanager
from scipy.optimize import curve_fit
from tqdm import tqdm

MSECS_PER_MINUTE = 60000


def plot_bleaching_curves(tc_times, tczyx_data, channel_names, output_file, title=''):
    """Plots bleaching curves and estimates bleaching lifetimes

    Parameters
    ----------
    tc_times : NDArray with shape (T, C)
        Times of acquisition for each time point and channel (minutes)
    tczyx_data : NDArray with shape (T, C, Z, Y, X)
        Raw data
    channel_names : list of strings with length (C)
    output_file : str
    title : str, optional
        plot title, by default ''
    """
    num_times = tczyx_data.shape[0]
    num_channels = tczyx_data.shape[1]

    means = np.zeros_like(tc_times)
    stds = np.zeros_like(tc_times)

    # Calculate statistics
    for t in tqdm(range(num_times)):
        for c in range(num_channels):
            zyx_data = tczyx_data[t, c, ...]  # zyx
            means[t, c] = np.mean(zyx_data)
            stds[t, c] = np.std(zyx_data)

    # Generate and save plots
    colors = ['g', 'r', 'b', 'c', 'm', 'k']

    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    for c in range(num_channels):
        xdata = tc_times[:, c]
        ydata = means[:, c]
        yerr = stds[:, c]

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
            ax.plot(xx, func(xx, *popt), color=colors[c], alpha=0.5)
            label = channel_names[c] + f" - {popt[1]:0.0f} minutes"
            print("Curve fit successful!")
            print(label)
        except:
            label = channel_names[c]
            print("Curve fit failed!")

        # Plot data
        ax.plot(
            xdata,
            ydata,
            label=label,
            marker='o',
            markeredgewidth=0,
            linewidth=0,
            color=colors[c],
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
@click.argument(
    "data_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output-folder",
    "-o",
    default=None,
    required=False,
    help="Path to output folder",
)
def estimate_bleaching(data_path, output_folder):
    """Estimate bleaching from raw data"""
    # Read data
    reader = read_micromanager(data_path)
    num_positions = reader.get_num_positions()

    # Handle paths
    input_folder = os.path.basename(os.path.normpath(data_path))
    if output_folder is None:
        output_folder = input_folder + "_bleaching"
    os.makedirs(output_folder, exist_ok=True)

    # Generate plot for each position
    for p in range(num_positions):
        print(f"Generating bleaching curves for position {p+1}/{num_positions}")

        tc_times = np.zeros((reader.shape[0], reader.shape[1]))
        for t in range(reader.shape[0]):
            for c in range(reader.shape[1]):
                try:
                    t0 = np.float32(reader.get_image_metadata(p, 0, c, 0)["TimeStampMsec"])
                    time = np.float32(reader.get_image_metadata(p, t, c, 0)["TimeStampMsec"])
                except:
                    print(f"WARNING: missing time metadata for p={p}, t={t}, c={c}")
                    t0 = np.nan
                    time = np.nan
                tc_times[t, c] = (time - t0) / MSECS_PER_MINUTE

        channel_names = [x.split(' ')[0] for x in reader.channel_names]
        tczyx_data = reader.get_zarr(p)
        output_file = os.path.join(output_folder, f"{p:03d}.svg")
        title = input_folder + f" - position = {p}"
        plot_bleaching_curves(tc_times, tczyx_data, channel_names, output_file, title)
