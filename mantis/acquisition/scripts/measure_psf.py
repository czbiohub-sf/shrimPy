# %%
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
from numpy.typing import ArrayLike
from scipy.ndimage import uniform_filter
from skimage.feature import peak_local_max

# %%

data_dir = Path(r'Z:\2022_12_22_LS_after_SL2')
dataset = 'epi_beads_100nm_fl_mount_after_SL2_1'

# data_path = data_dir / dataset / (dataset+'_MMStack_Pos0.ome.tif')
data_path = r'Z:\2022_12_22_LS_after_SL2\epi_beads_100nm_fl_mount_after_SL2_1\LS_beads_100nm_fl_mount_after_SL2_1_MMStack_Pos0.ome.tif'
zyx_data = tifffile.imread(data_path)

scale = (0.250, 0.069, 0.069)  # in um
axis_labels = ("Z", "Y", "X")

# %%

# runs in about 10 seconds, sensitive to parameters
# finds ~310 peaks
points = peak_local_max(
    uniform_filter(zyx_data, size=3),  # helps remove hot pixels, adds ~3s
    min_distance=25,
    threshold_abs=200,
    num_peaks=1000,  # limit to top 1000 peaks
    exclude_border=(3, 10, 10),  # in zyx
)

# %%
viewer = napari.Viewer()
viewer.add_image(zyx_data)

viewer.add_points(points, name='peaks local max', size=12, symbol='ring', edge_color='yellow')

# %%

patch_size = (scale[0] * 11, scale[1] * 15, scale[2] * 15)
# round to nearest 0.5 um
# patch_size = np.round(np.asarray(patch_size) / 0.5) * 0.5

# extract bead patches
bead_extractor = BeadExtractor(
    image=Calibrated3DImage(data=zyx_data.astype(np.float64), spacing=scale),
    patch_size=patch_size,
)
beads = bead_extractor.extract_beads(points=points)
bead_offsets = np.asarray([bead.offset for bead in beads])


def analyze_psf(bead: Calibrated3DImage):
    psf = PSF(image=bead)
    try:
        psf.analyze()
        return psf.get_summary_dict()
    except Exception:
        # skip over beads where psf analysis failed
        return {}


# analyze bead patches
results = [analyze_psf(bead) for bead in beads]
num_failed = sum([result == {} for result in results])

df = pd.DataFrame.from_records(results)
df['z_mu'] += bead_offsets[:, 0] * scale[0]
df['y_mu'] += bead_offsets[:, 1] * scale[1]
df['x_mu'] += bead_offsets[:, 2] * scale[2]
df = df.dropna()

# %%


def visualize_psf(out_dir: str, zyx_data: ArrayLike, zyx_scale: tuple, axis_labels: tuple):
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = zyx_data.shape
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(
        zyx_data[shape_Z // 2, :, :], cmap=cmap, origin='lower', aspect=scale_Y / scale_X
    )
    ax[0].set_xlabel(axis_labels[-1])
    ax[0].set_ylabel(axis_labels[-2])

    ax[1].imshow(
        zyx_data[:, shape_Y // 2, :], cmap=cmap, origin='lower', aspect=scale_Z / scale_X
    )
    ax[1].set_xlabel(axis_labels[-1])
    ax[1].set_ylabel(axis_labels[-3])

    ax[2].imshow(
        zyx_data[:, :, shape_X // 2], cmap=cmap, origin='lower', aspect=scale_Z / scale_Y
    )
    ax[2].set_xlabel(axis_labels[-2])
    ax[2].set_ylabel(axis_labels[-3])

    for _ax in ax.flatten():
        _ax.set_xticks([])
        _ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(left=0.12, wspace=0.5)
    fig_size = fig.get_size_inches()
    fig_size_scaling = 3 / fig_size[0]  # set width to 3 inches
    fig.set_figwidth(fig_size[0] * fig_size_scaling)
    fig.set_figheight(fig_size[1] * fig_size_scaling)
    fig.savefig(out_dir)


def plot_fwhm_vs_z(out_dir: str, z, fwhm_x, fwhm_y, fwhm_z):
    fig, ax = plt.subplots(1, 1)

    ax.plot(z, fwhm_x, 'o', z, fwhm_y, 'o')
    ax.set_ylabel('X and Y FWHM (um)')
    ax.set_xlabel('Z position (um)')

    ax2 = ax.twinx()
    ax2.plot(z, fwhm_z, 'o', color='green')
    ax2.set_ylabel('Z FWHM (um)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.savefig(out_dir)


# %% generate plots

plots_dir = data_dir / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)

random_bead_number = np.random.choice(len(df), 3)
bead_psf_paths = [''] * 3
for i, path, bead_number in zip(range(3), bead_psf_paths, random_bead_number):
    path = plots_dir / f'bead_psf_{i}.png'
    bead_psf_paths[i] = path
    visualize_psf(path, beads[bead_number].data, scale, axis_labels)

fwhm_vs_z_3d_path = plots_dir / 'fwhm_vs_z_3d.png'
plot_fwhm_vs_z(
    fwhm_vs_z_3d_path,
    df['z_mu'].values,
    *[df[col].values for col in ('zyx_x_fwhm', 'zyx_y_fwhm', 'zyx_z_fwhm')],
)

fwhm_vs_z_1d_path = plots_dir / 'fwhm_vs_z_1d.png'
plot_fwhm_vs_z(
    fwhm_vs_z_1d_path,
    df['z_mu'].values,
    *[df[col].values for col in ('x_fwhm', 'y_fwhm', 'z_fwhm')],
)

mean_1d_fwhm = [df[col].mean() for col in ('x_fwhm', 'y_fwhm', 'z_fwhm')]
mean_3d_fwhm = [df[col].mean() for col in ('zyx_x_fwhm', 'zyx_y_fwhm', 'zyx_z_fwhm')]
mean_pc_fwhm = [df[col].mean() for col in ('zyx_pc3_fwhm', 'zyx_pc2_fwhm', 'zyx_pc1_fwhm')]

# %%

# generate html report
html = markdown.markdown(
    f'''
# PSF Analysis

## Overview
Dataset name: {dataset}

Scale: {scale}

Mean FWHM:

* 1D: {'({:.3f}, {:.3f}, {:.3f})'.format(*mean_1d_fwhm)}
* 3D: {'({:.3f}, {:.3f}, {:.3f})'.format(*mean_3d_fwhm)}
* PC: {'({:.3f}, {:.3f}, {:.3f})'.format(*mean_pc_fwhm)}

## Representative bead PSF images
![bead psf 1]({bead_psf_paths[0]} "bead psf {random_bead_number[0]}")
![bead psf 2]({bead_psf_paths[1]} "bead psf {random_bead_number[1]}")
![bead psf 3]({bead_psf_paths[2]} "bead psf {random_bead_number[2]}")

## XYZ FWHM versus Z position (3D)
![fwhm vs z]({fwhm_vs_z_3d_path} "fwhm vs z")

'''
)

# %%

# save html file and show in browser
html_file_path = Path(r'C:\Users\labelfree\Documents\temphtml.html')
with open(html_file_path, 'w') as file:
    file.write(html)

webbrowser.open('file://' + str(html_file_path))

# %%
