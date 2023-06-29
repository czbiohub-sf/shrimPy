import os

import click
import matplotlib.pyplot as plt
import numpy as np

from iohub.ngff import open_ome_zarr, Plate
from scipy.optimize import curve_fit
from tqdm import tqdm

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
        for c in range(num_channels):
            zyx_data = tczyx_data[t, c, ...]  # zyx
            means[t, c] = np.mean(zyx_data)
            stds[t, c] = np.std(zyx_data)

    # Generate and save plots
    colors = ['g', 'r', 'b', 'c', 'm', 'k']

    f, ax = plt.subplots(1, 1, figsize=(4, 4))
    for c in range(num_channels):
        xdata = times[:]
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
        except Exception as e:
            print(e)
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
    "input_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output-folder",
    "-o",
    default=None,
    required=False,
    help="Path to output folder",
)
def estimate_bleaching(input_path, output_folder):
    """Estimate bleaching from raw data"""
    reader = open_ome_zarr(input_path)
    if not isinstance(reader, Plate):
        raise ValueError("Please supply an HCS plate .zarr store.")

    # Handle paths
    input_folder = os.path.splitext(os.path.basename(os.path.normpath(input_path)))[0]
    if output_folder is None:
        output_folder = input_folder + "_bleaching"

    # Read timing metadata and generate plots
    for well_name, position in reader.positions():
        print(f"Generating bleaching curves for position {well_name}")

        # Generate plot for each position
        T = position.data.shape[0]
        try:
            dt = np.float32(reader.zattrs['Summary']['Interval_ms'] / MSECS_PER_MINUTE)
        except Exception as e:
            print(e)
            print(f"WARNING: missing time metadata for p={p}, t={t}, c={c}")
            dt = 1

        times = np.arange(0, T * dt, step=dt)
        tczyx_data = position.data
        output_file = os.path.join(output_folder, well_name)
        os.makedirs(output_file, exist_ok=True)
        title = input_folder + f" - position = {well_name}"
        plot_bleaching_curves(
            times,
            tczyx_data,
            reader.channel_names,
            os.path.join(output_file, "bleaching.svg"),
            title,
        )
