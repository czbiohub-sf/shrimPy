import datetime
import pickle
import shutil
import webbrowser

from pathlib import Path
from typing import List

import markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.psf import PSF
from numpy.typing import ArrayLike
from scipy.signal import peak_widths


def _make_plots(
    output_path: Path,
    beads: List[ArrayLike],
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple,
    axis_labels: tuple,
    raw: bool = False,
):
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    random_bead_number = sorted(np.random.choice(len(beads), 3))

    bead_psf_slices_paths = plot_psf_slices(
        plots_dir,
        [beads[i] for i in random_bead_number],
        scale,
        axis_labels,
        random_bead_number,
    )

    if raw:
        plot_data_x = [df_1d_peak_width[col].values for col in ('x_mu', 'y_mu', 'z_mu')]
        plot_data_y = [
            df_1d_peak_width[col].values for col in ('1d_x_fwhm', '1d_y_fwhm', '1d_z_fwhm')
        ]
    else:
        plot_data_x = [df_gaussian_fit[col].values for col in ('x_mu', 'y_mu', 'z_mu')]
        plot_data_y = [
            df_gaussian_fit[col].values for col in ('zyx_x_fwhm', 'zyx_y_fwhm', 'zyx_z_fwhm')
        ]

    fwhm_vs_acq_axes_paths = plot_fwhm_vs_acq_axes(
        plots_dir,
        *plot_data_x,
        *plot_data_y,
        axis_labels,
    )

    psf_amp_paths = plot_psf_amp(
        plots_dir,
        df_gaussian_fit['x_mu'].values,
        df_gaussian_fit['y_mu'].values,
        df_gaussian_fit['z_mu'].values,
        df_gaussian_fit['zyx_amp'].values,
        axis_labels,
    )

    return (bead_psf_slices_paths, fwhm_vs_acq_axes_paths, psf_amp_paths)


def generate_report(
    output_path: Path,
    data_dir: Path,
    dataset: str,
    beads: List[ArrayLike],
    peaks: ArrayLike,
    df_gaussian_fit: pd.DataFrame,
    df_1d_peak_width: pd.DataFrame,
    scale: tuple,
    axis_labels: tuple,
):
    output_path.mkdir(exist_ok=True)

    num_beads = len(beads)
    num_successful = len(df_gaussian_fit)
    num_failed = num_beads - num_successful

    raw = False
    if axis_labels == ("SCAN", "TILT", "COVERSLIP"):
        raw = True

    # make plots
    (bead_psf_slices_paths, fwhm_vs_acq_axes_paths, psf_amp_paths) = _make_plots(
        output_path, beads, df_gaussian_fit, df_1d_peak_width, scale, axis_labels, raw=raw
    )

    # calculate statistics
    fwhm_3d_mean = [
        df_gaussian_fit[col].mean() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')
    ]
    fwhm_3d_std = [
        df_gaussian_fit[col].std() for col in ('zyx_z_fwhm', 'zyx_y_fwhm', 'zyx_x_fwhm')
    ]
    fwhm_pc_mean = [
        df_gaussian_fit[col].mean() for col in ('zyx_pc3_fwhm', 'zyx_pc2_fwhm', 'zyx_pc1_fwhm')
    ]
    fwhm_1d_mean = df_1d_peak_width.mean()
    fwhm_1d_std = df_1d_peak_width.std()

    # generate html report
    html_report = _generate_html(
        dataset,
        data_dir,
        scale,
        (num_beads, num_successful, num_failed),
        fwhm_1d_mean,
        fwhm_1d_std,
        fwhm_3d_mean,
        fwhm_3d_std,
        fwhm_pc_mean,
        bead_psf_slices_paths,
        fwhm_vs_acq_axes_paths,
        psf_amp_paths,
        axis_labels,
    )

    # save html report and other results
    with open(output_path / 'peaks.pkl', 'wb') as file:
        pickle.dump(peaks, file)

    df_gaussian_fit.to_csv(output_path / 'psf_gaussian_fit.csv', index=False)
    df_1d_peak_width.to_csv(output_path / 'psf_1d_peak_width.csv', index=False)

    shutil.copy('github-markdown.css', output_path)
    html_file_path = output_path / ('psf_analysis_report.html')
    with open(html_file_path, 'w') as file:
        file.write(html_report)

    # display html report
    webbrowser.open('file://' + str(html_file_path))


def extract_beads(zyx_data: ArrayLike, points: ArrayLike, scale: tuple):
    patch_size = (scale[0] * 15, scale[1] * 18, scale[2] * 18)

    # extract bead patches
    bead_extractor = BeadExtractor(
        image=Calibrated3DImage(data=zyx_data.astype(np.uint16), spacing=scale),
        patch_size=patch_size,
    )
    beads = bead_extractor.extract_beads(points=points)
    # remove bad beads
    beads = [bead for bead in beads if bead.data.size > 0]
    beads_data = [bead.data for bead in beads]
    bead_offset = [bead.offset for bead in beads]

    return beads_data, bead_offset


def analyze_psf(zyx_patches: List[ArrayLike], bead_offsets: List[tuple], scale: tuple):
    results = []
    for patch, offset in zip(zyx_patches, bead_offsets):
        bead = Calibrated3DImage(data=patch.astype(np.uint16), spacing=scale, offset=offset)
        psf = PSF(image=bead)
        try:
            psf.analyze()
            summary_dict = psf.get_summary_dict()
        except Exception:
            summary_dict = {}
        results.append(summary_dict)

    df_gaussian_fit = pd.DataFrame.from_records(results)
    bead_offsets = np.asarray(bead_offsets)

    df_gaussian_fit['z_mu'] += bead_offsets[:, 0] * scale[0]
    df_gaussian_fit['y_mu'] += bead_offsets[:, 1] * scale[1]
    df_gaussian_fit['x_mu'] += bead_offsets[:, 2] * scale[2]

    df_1d_peak_width = pd.DataFrame(
        [calculate_peak_widths(zyx_patch, scale) for zyx_patch in zyx_patches],
        columns=(f'1d_{i}_fwhm' for i in ('z', 'y', 'x')),
    )
    df_1d_peak_width = pd.concat(
        (df_gaussian_fit[['z_mu', 'y_mu', 'x_mu']], df_1d_peak_width), axis=1
    )

    # clean up dataframes
    df_gaussian_fit = df_gaussian_fit.dropna()
    df_1d_peak_width = df_1d_peak_width.loc[
        ~(df_1d_peak_width[['1d_z_fwhm', '1d_y_fwhm', '1d_x_fwhm']] == 0).any(axis=1)
    ]

    return df_gaussian_fit, df_1d_peak_width


def calculate_peak_widths(zyx_data: ArrayLike, zyx_scale: tuple):
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = zyx_data.shape

    try:
        z_fwhm = peak_widths(zyx_data[:, shape_Y // 2, shape_X // 2], [shape_Z // 2])[0][0]
        y_fwhm = peak_widths(zyx_data[shape_Z // 2, :, shape_X // 2], [shape_Y // 2])[0][0]
        x_fwhm = peak_widths(zyx_data[shape_Z // 2, shape_Y // 2, :], [shape_X // 2])[0][0]
    except Exception:
        z_fwhm, y_fwhm, x_fwhm = (0.0, 0.0, 0.0)

    return z_fwhm * scale_Z, y_fwhm * scale_Y, x_fwhm * scale_X


def _adjust_fig(fig, ax):
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
    plots_dir: str,
    beads: List[ArrayLike],
    zyx_scale: tuple,
    axis_labels: tuple,
    bead_numbers: list,
):
    num_beads = len(beads)
    scale_Z, scale_Y, scale_X = zyx_scale
    shape_Z, shape_Y, shape_X = beads[0].shape
    cmap = 'viridis'

    bead_xy_psf_path = plots_dir / 'beads_xy_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead, bead_number in zip(ax, beads, bead_numbers):
        _ax.imshow(
            bead[shape_Z // 2, :, :], cmap=cmap, origin='lower', aspect=scale_Y / scale_X
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-2])
        _ax.set_title(f'Bead: {bead_number}')
    _adjust_fig(fig, ax)
    fig.set_figheight(2)
    fig.savefig(bead_xy_psf_path)

    bead_xz_psf_path = plots_dir / 'beads_xz_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead in zip(ax, beads):
        _ax.imshow(
            bead[:, shape_Y // 2, :], cmap=cmap, origin='lower', aspect=scale_Z / scale_X
        )
        _ax.set_xlabel(axis_labels[-1])
        _ax.set_ylabel(axis_labels[-3])
    _adjust_fig(fig, ax)
    fig.savefig(bead_xz_psf_path)

    bead_yz_psf_path = plots_dir / 'beads_yz_psf.png'
    fig, ax = plt.subplots(1, num_beads)
    for _ax, bead in zip(ax, beads):
        _ax.imshow(
            bead[:, :, shape_X // 2], cmap=cmap, origin='lower', aspect=scale_Z / scale_Y
        )
        _ax.set_xlabel(axis_labels[-2])
        _ax.set_ylabel(axis_labels[-3])
    _adjust_fig(fig, ax)
    fig.savefig(bead_yz_psf_path)

    return bead_xy_psf_path, bead_xz_psf_path, bead_yz_psf_path


def plot_fwhm_vs_acq_axes(plots_dir: str, x, y, z, fwhm_x, fwhm_y, fwhm_z, axis_labels: tuple):
    def plot_fwhm_vs_acq_axis(out_dir: str, x, fwhm_x, fwhm_y, fwhm_z, x_axis_label: str):
        fig, ax = plt.subplots(1, 1)
        artist1 = ax.plot(x, fwhm_x, 'o', x, fwhm_y, 'o')
        ax.set_ylabel('{} and {} FWHM (um)'.format(*axis_labels[1:][::-1]))
        ax.set_xlabel('{} position (um)'.format(x_axis_label))

        ax2 = ax.twinx()
        artist2 = ax2.plot(x, fwhm_z, 'o', color='green')
        ax2.set_ylabel('{} FWHM (um)'.format(axis_labels[0]), color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        plt.legend(artist1 + artist2, axis_labels[::-1])
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


def _generate_html(
    dataset_name: str,
    data_path: str,
    dataset_scale: tuple,
    num_beads_total_good_bad: tuple,
    fwhm_1d_mean: tuple,
    fwhm_1d_std: tuple,
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

* **3D Gaussian fit**
    - {axis_labels[-1]}: {fwhm_3d_mean[-1]:.3f} ± {fwhm_3d_std[0]:.3f} um
    - {axis_labels[-2]}: {fwhm_3d_mean[-2]:.3f} ± {fwhm_3d_std[1]:.3f} um
    - {axis_labels[-3]}: {fwhm_3d_mean[-3]:.3f} ± {fwhm_3d_std[2]:.3f} um
* 1D profile
    - {axis_labels[-1]}: {fwhm_1d_mean[-1]:.3f} ± {fwhm_1d_std[0]:.3f} um
    - {axis_labels[-2]}: {fwhm_1d_mean[-2]:.3f} ± {fwhm_1d_std[1]:.3f} um
    - {axis_labels[-3]}: {fwhm_1d_mean[-3]:.3f} ± {fwhm_1d_std[2]:.3f} um
* 3D principal components
    - {'{:.3f} um, {:.3f} um, {:.3f} um'.format(*fwhm_pc_mean)}

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


def detect_peaks(
    zyx_data: np.ndarray,
    block_size: int | tuple[int, int, int] = (128, 64, 64),
    min_distance: int = 40,
    threshold_abs: float = 200.0,
    max_num_peaks: int = 2000,
    exclude_border: tuple[int, int, int] | None = None,
    blur_kernel_size: int = 3,
    device: str = "cpu",
):
    """Detect peaks with local maxima.
    This is an approximate torch implementation of `skimage.feature.peak_local_max`.

    Parameters
    ----------
    zyx_data : np.ndarray
        3D image data
    block_size : int | tuple[int, int, int], optional
        block size to find approximate local maxima, by default (128, 64, 64)
    min_distance : int, optional
        minimum distance between detections,
        this is calculated assuming a Cartesian coordinate system,
        needs to be smaller than block size for efficiency,
        by default 40
    threshold_abs : float, optional
        lower bound of detected peak intensity, by default 200.0
    max_num_peaks : int, optional
        max number of candidate detections to consider, by default 2000
    exclude_border : tuple[int, int, int] | None, optional
        width of borders to exclude, by default None
    blur_kernel_size : int, optional
        uniform kernel size to blur the image before detection
        to avoid hot pixels, by default 3
    device : str, optional
        compute device string for torch,
        e.g. "cpu" (slow), "cuda" (single GPU) or "cuda:0" (0th GPU among multiple),
        by default "cpu"

    Returns
    -------
    np.ndarray
        3D coordinates of detected peaks (N, 3)

    """
    zyx_image = torch.from_numpy(zyx_data)
    if device != "cpu":
        zyx_image = zyx_image.to(torch.float16).to(device)

    if blur_kernel_size % 2 != 1:
        raise ValueError(f"kernel_size={blur_kernel_size} must be an odd number")
    zyx_image = F.avg_pool3d(
        input=zyx_image[None, None],
        kernel_size=blur_kernel_size,
        stride=1,
        padding=blur_kernel_size // 2,
        count_include_pad=False,
    )[0, 0]

    input_shape = zyx_image.shape
    image = zyx_image[None, None]
    peak_value, peak_idx = (
        p.flatten().clone()
        for p in F.max_pool3d(
            image,
            kernel_size=block_size,
            stride=block_size,
            padding=0,
            return_indices=True,
        )
    )
    peak_value, sort_mask = peak_value.topk(min(max_num_peaks, peak_value.nelement()))
    peak_idx = peak_idx[sort_mask]
    abs_mask = peak_value > threshold_abs
    peak_value = peak_value[abs_mask]
    peak_idx = peak_idx[abs_mask]
    # requires torch>=2.2
    coords = torch.stack(torch.unravel_index(peak_idx, input_shape), -1)
    fcoords = coords.float()
    dist = torch.cdist(fcoords, fcoords)
    dist_mask = (dist < min_distance).sum(0) < 2
    match exclude_border:
        case None:
            pass
        case (int(), int(), int()):
            for dim, size in enumerate(exclude_border):
                border_mask = (size < coords[:, dim]) & (
                    coords[:, dim] < input_shape[dim] - size
                )
                dist_mask &= border_mask
        case _:
            raise ValueError(f"invalud argument exclude_border={exclude_border}")
    return coords[dist_mask].cpu().numpy()
