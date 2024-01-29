# %%
import datetime
import pickle
import shutil
import webbrowser

from pathlib import Path

import markdown
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.psf import PSF
from scipy.ndimage import uniform_filter
from skimage.feature import peak_local_max


def analyze_psf(bead: Calibrated3DImage):
    psf = PSF(image=bead)
    try:
        psf.analyze()
        return psf.get_summary_dict()
    except Exception:
        # skip over beads where psf analysis failed
        return {}


def adjust_fig(fig, ax):
    for _ax in ax.flatten():
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5)
    fig_size = fig.get_size_inches()
    fig_size_scaling = 5 / fig_size[0]  # set width to 5 inches
    fig.set_figwidth(fig_size[0] * fig_size_scaling)
    fig.set_figheight(fig_size[1] * fig_size_scaling)


def plot_psf_slices(
    plots_dir: str, beads: list, zyx_scale: tuple, axis_labels: tuple, bead_numbers: list
):
    num_beads = len(beads)
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = beads[0].data.shape
    cmap = 'viridis'

    bead_xy_psf_path = plots_dir / 'beads_xy_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead, bead_number in zip(ax, beads, bead_numbers):
        _ax.imshow(
            bead.data[shape_Z // 2, :, :], cmap=cmap, origin='lower', aspect=scale_Y / scale_X
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-2])
        _ax.set_title(f'Bead: {bead_number}')
    adjust_fig(fig, ax)
    fig.set_figheight(2)
    fig.savefig(bead_xy_psf_path)

    bead_xz_psf_path = plots_dir / 'beads_xz_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead in zip(ax, beads):
        _ax.imshow(
            bead.data[:, shape_Y // 2, :], cmap=cmap, origin='lower', aspect=scale_Z / scale_X
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-3])
    adjust_fig(fig, ax)
    fig.savefig(bead_xz_psf_path)

    bead_yz_psf_path = plots_dir / 'beads_yz_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead in zip(ax, beads):
        _ax.imshow(
            bead.data[:, :, shape_X // 2], cmap=cmap, origin='lower', aspect=scale_Z / scale_Y
        )
        _ax.set_xlabel(axis_labels[-2])
        _ax.set_ylabel(axis_labels[-3])
    adjust_fig(fig, ax)
    fig.savefig(bead_yz_psf_path)

    return bead_xy_psf_path, bead_xz_psf_path, bead_yz_psf_path


def plot_fwhm_vs_acq_axes(plots_dir: str, x, y, z, fwhm_x, fwhm_y, fwhm_z, axis_labels: tuple):
    def plot_fwhm_vs_acq_axis(out_dir: str, x, fwhm_x, fwhm_y, fwhm_z, x_axis_label: str):
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, fwhm_x, 'o', x, fwhm_y, 'o')
        ax.set_ylabel('{} and {} FWHM (um)'.format(*axis_labels[1:][::-1]))
        ax.set_xlabel('{} position (um)'.format(x_axis_label))

        ax2 = ax.twinx()
        ax2.plot(x, fwhm_z, 'o', color='green')
        ax2.set_ylabel('{} FWHM (um)'.format(axis_labels[0]), color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        fig.savefig(out_dir)

    out_dirs = [plots_dir / f'fwhm_vs_{axis}.png' for axis in axis_labels]
    for our_dir, x_axis, x_axis_label in zip(out_dirs, (z, y, x), axis_labels):
        plot_fwhm_vs_acq_axis(our_dir, x_axis, fwhm_x, fwhm_y, fwhm_z, x_axis_label)

    return out_dirs


def plot_psf_amp(plots_dir: str, x, y, z, amp, axis_labels: tuple):
    psf_amp_xy_path = plots_dir / 'psf_amp_xy.png'
    fig, ax = plt.subplots(1, 1)

    sc = ax.scatter(
        x,
        y,
        c=amp,
        vmin=np.quantile(amp, 0.01),
        vmax=np.quantile(amp, 0.99),
        cmap='summer',
    )
    ax.set_aspect('equal')
    ax.set_xlabel(f'{axis_labels[-1]} (um)')
    ax.set_ylabel(f'{axis_labels[-2]} (um)')
    plt.colorbar(sc, label='Amplitude (a.u.)')
    fig.savefig(psf_amp_xy_path)

    psf_amp_z_path = plots_dir / 'psf_amp_z.png'
    fig, ax = plt.subplots(1, 1)
    ax.scatter(z, amp)
    ax.set_xlabel(f'{axis_labels[-3]} (um)')
    ax.set_ylabel('Amplitude (a.u.)')
    fig.savefig(psf_amp_z_path)

    return psf_amp_xy_path, psf_amp_z_path


def generate_html_report(
    dataset_name: str,
    data_path: str,
    dataset_scale: tuple,
    num_beads_total_good_bad: tuple,
    fwhm_3d_mean: tuple,
    fwhm_3d_std: tuple,
    fwhm_pc_mean: tuple,
    bead_psf_slices_paths: tuple,
    fwhm_vs_acq_axes_paths: tuple,
    psf_amp_paths: tuple,
    axis_labels: tuple,
):

    # string indents need to be like that, otherwise this turns into a code block
    report_str = f'''
# PSF Analysis

## Overview

### Dataset

* Name: `{dataset_name}`
* Path: `{data_path}`
* Scale: {dataset_scale[::-1]} um  <!-- in XYZ order -->
* Date analyzed: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Number of beads

* Defected: {num_beads_total_good_bad[0]}
* Analyzed: {num_beads_total_good_bad[1]}
* Skipped: {num_beads_total_good_bad[2]}

### FWHM

* {axis_labels[-1]}: {fwhm_3d_mean[-1]:.3f} ± {fwhm_3d_std[0]:.3f} um
* {axis_labels[-2]}: {fwhm_3d_mean[-2]:.3f} ± {fwhm_3d_std[1]:.3f} um
* {axis_labels[-3]}: {fwhm_3d_mean[-2]:.3f} ± {fwhm_3d_std[2]:.3f} um
* PC: {'{:.3f} um, {:.3f} um, {:.3f} um'.format(*fwhm_pc_mean)}

## Representative bead PSF images
![beads xy psf]({bead_psf_slices_paths[0]})
![beads xz psf]({bead_psf_slices_paths[1]})
![beads yz psf]({bead_psf_slices_paths[2]})

## FWHM versus {axis_labels[0]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[0]} "fwhm vs z")

## FWHM versus {axis_labels[1]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[1]} "fwhm vs y")

## FWHM versus {axis_labels[2]} position
![fwhm vs z]({fwhm_vs_acq_axes_paths[2]} "fwhm vs x")

## PSF amplitude versus {axis_labels[-1]}-{axis_labels[-2]} position
![psf amp xy]({psf_amp_paths[0]} "psf amp xy")

## PSF amplitude versus {axis_labels[-3]} position
![psf amp z]({psf_amp_paths[1]} "psf amp z")
'''

    css_style = '''
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="github-markdown.css">
<style>
    .markdown-body {
        box-sizing: border-box;
        min-width: 200px;
        max-width: 980px;
        margin: 0 auto;
        padding: 45px;
    }

    @media (max-width: 767px) {
        .markdown-body {
            padding: 15px;
        }
    }
</style>
'''

    html = markdown.markdown(report_str)
    formatted_html = f'''
{css_style}
<article class="markdown-body">
{html}
</article>
'''.strip()

    return formatted_html


# %% Load data - swap with data acquisition block

data_dir = Path('/Users/ivan.ivanov/Documents/images_local/')
dataset = 'epi_beads_100nm_fl_mount_after_SL2_1'

# data_path = data_dir / dataset / (dataset+'_MMStack_Pos0.ome.tif')
psf_analysis_path = data_dir / dataset / 'psf_analysis'
data_path = data_dir / dataset / 'LS_beads_100nm_fl_mount_after_SL2_1_MMStack_Pos0.ome.tif'
zyx_data = tifffile.imread(data_path)

scale = (0.250, 0.069, 0.069)  # in um
axis_labels = ("Z", "Y", "X")

psf_analysis_path.mkdir(exist_ok=True)

# %% Find peaks

# runs in about 10 seconds, sensitive to parameters
# finds ~310 peaks
points = peak_local_max(
    uniform_filter(zyx_data, size=3),  # helps remove hot pixels, adds ~3s
    min_distance=25,
    threshold_abs=200,
    num_peaks=1000,  # limit to top 1000 peaks
    exclude_border=(3, 10, 10),  # in zyx
)

# %% Visualize in napari

viewer = napari.Viewer()
viewer.add_image(zyx_data)

viewer.add_points(points, name='peaks local max', size=12, symbol='ring', edge_color='yellow')

# %% Extract and analyze bead patches

patch_size = (scale[0] * 11, scale[1] * 15, scale[2] * 15)

# extract bead patches
bead_extractor = BeadExtractor(
    image=Calibrated3DImage(data=zyx_data.astype(np.float64), spacing=scale),
    patch_size=patch_size,
)
beads = bead_extractor.extract_beads(points=points)
bead_offsets = np.asarray([bead.offset for bead in beads])

# analyze bead patches
num_beads = len(beads)
results = [analyze_psf(bead) for bead in beads]
num_failed = sum([result == {} for result in results])
num_successful = num_beads - num_failed

df = pd.DataFrame.from_records(results)
df['z_mu'] += bead_offsets[:, 0] * scale[0]
df['y_mu'] += bead_offsets[:, 1] * scale[1]
df['x_mu'] += bead_offsets[:, 2] * scale[2]
df = df.dropna()

# %% Generate plots

plots_dir = psf_analysis_path / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
random_bead_number = sorted(np.random.choice(len(df), 3))

bead_psf_slices_paths = plot_psf_slices(
    plots_dir, [beads[i] for i in random_bead_number], scale, axis_labels, random_bead_number
)

fwhm_vs_acq_axes_paths = plot_fwhm_vs_acq_axes(
    plots_dir,
    df['x_mu'].values,
    df['y_mu'].values,
    df['z_mu'].values,
    *[df[col].values for col in ('zyx_x_fwhm', 'zyx_y_fwhm', 'zyx_z_fwhm')],
    axis_labels,
)

psf_amp_paths = plot_psf_amp(
    plots_dir,
    df['x_mu'].values,
    df['y_mu'].values,
    df['z_mu'].values,
    df['zyx_amp'].values,
    axis_labels,
)

# mean_1d_fwhm = [df[col].mean() for col in ('x_fwhm', 'y_fwhm', 'z_fwhm')]
fwhm_3d_mean = [df[col].mean() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')]
fwhm_3d_std = [df[col].std() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')]
fwhm_pc_mean = [df[col].mean() for col in ('zyx_pc3_fwhm', 'zyx_pc2_fwhm', 'zyx_pc1_fwhm')]

# %% Generate HTML report

html_report = generate_html_report(
    dataset,
    data_path.parent,
    scale,
    (num_beads, num_successful, num_failed),
    fwhm_3d_mean,
    fwhm_3d_std,
    fwhm_pc_mean,
    bead_psf_slices_paths,
    fwhm_vs_acq_axes_paths,
    psf_amp_paths,
    axis_labels,
)

# save html file and show in browser
with open(psf_analysis_path / 'peaks.pkl', 'wb') as file:
    pickle.dump(points, file)

df.to_csv(psf_analysis_path / 'psf_analysis.csv', index=False)

shutil.copy('github-markdown.css', psf_analysis_path)
html_file_path = psf_analysis_path / ('psf_analysis_report.html')
with open(html_file_path, 'w') as file:
    file.write(html_report)

webbrowser.open('file://' + str(html_file_path))

# %%
