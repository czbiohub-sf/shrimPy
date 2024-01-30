# %%
import pickle
import shutil
import webbrowser

from pathlib import Path

import napari
import numpy as np

from iohub.reader import open_ome_zarr, read_micromanager

from mantis.analysis.analyze_psf import (
    analyze_psf,
    detect_peaks,
    generate_html_report,
    plot_fwhm_vs_acq_axes,
    plot_psf_amp,
    plot_psf_slices,
)

# %% Load data - swap with data acquisition block

data_dir = Path(r'Z:\2023_03_30_beads')
dataset = 'beads_ip_0.74_1'

# data_dir = Path(r'Z:\2022_12_22_LS_after_SL2')
# dataset = 'epi_beads_100nm_fl_mount_after_SL2_1'
# data_path = data_dir / dataset / 'LS_beads_100nm_fl_mount_after_SL2_1_MMStack_Pos0.ome.tif'
# zyx_data = tifffile.imread(data_path)

data_path = data_dir / dataset

if str(data_path).endswith('.zarr'):
    ds = open_ome_zarr(data_path / '0/0/0')
    zyx_data = ds.data[0, 0]
else:
    ds = read_micromanager(str(data_path))
    zyx_data = ds.get_array(0)[0, 0]

scale = (0.1565, 0.116, 0.116)  # in um
# axis_labels = ("Z", "Y", "X")
axis_labels = ("SCAN", "TILT", "COVERSLIP")

# %% Detect peaks

raw = False
if axis_labels == ("SCAN", "TILT", "COVERSLIP"):
    raw = True

peaks = detect_peaks(zyx_data, raw=raw)
print(f'Number of peaks detected: {len(peaks)}')

# %% Visualize in napari

viewer = napari.Viewer()
viewer.add_image(zyx_data)

viewer.add_points(peaks, name='peaks local max', size=12, symbol='ring', edge_color='yellow')

# %% Extract and analyze bead patches

beads, df_gaussian_fit, df_1d_peak_width = analyze_psf(
    zyx_data=zyx_data,
    points=peaks,
    scale=scale,
)

# analyze bead patches
num_beads = len(beads)
num_successful = len(df_gaussian_fit)
num_failed = num_beads - num_successful

# %% Generate plots

psf_analysis_path = data_dir / dataset / 'psf_analysis'
psf_analysis_path.mkdir(exist_ok=True)

plots_dir = psf_analysis_path / 'plots'
plots_dir.mkdir(parents=True, exist_ok=True)
random_bead_number = sorted(np.random.choice(num_successful, 3))

bead_psf_slices_paths = plot_psf_slices(
    plots_dir, [beads[i] for i in random_bead_number], scale, axis_labels, random_bead_number
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

# %% Generate HTML report

html_report = generate_html_report(
    dataset,
    data_path.parent,
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

# save html file and show in browser
with open(psf_analysis_path / 'peaks.pkl', 'wb') as file:
    pickle.dump(peaks, file)

df_gaussian_fit.to_csv(psf_analysis_path / 'psf_gaussian_fit.csv', index=False)
df_1d_peak_width.to_csv(psf_analysis_path / 'psf_1d_peak_width.csv', index=False)

shutil.copy('github-markdown.css', psf_analysis_path)
html_file_path = psf_analysis_path / ('psf_analysis_report.html')
with open(html_file_path, 'w') as file:
    file.write(html_report)

webbrowser.open('file://' + str(html_file_path))

# %%
