# %% This script calculates the average column translation and row translation
# for image stitching calculated from the phase cross-correlation on overlapping tiles

import os

from pathlib import Path

import numpy as np

from iohub import open_ome_zarr
from skimage.registration import phase_cross_correlation

os.environ["DISPLAY"] = ':1005'

# %%
data_dir = Path(
    '/hpc/projects/intracellular_dashboard/ops/2024_03_05_registration_test/kidney_tissue/0-convert/dragonfly/'
)
dataset = 'kidney_grid_test_1.zarr'
data_path = data_dir / dataset

fliplr = True
flipud = False

rows_limit = 3
cols_limit = 3
percent_overlap = 0.05

dataset = open_ome_zarr(data_path)

grid_rows = set()
grid_cols = set()
well_name, well = next(dataset.wells())
for position_name, position in well.positions():
    fov_name = Path(position_name).parts[-1]
    grid_rows.add(fov_name[3:])  # 1-Pos<COL>_<ROW> syntax
    grid_cols.add(fov_name[:3])
sizeY, sizeX = position.data.shape[-2:]
grid_rows = sorted(grid_rows)
grid_cols = sorted(grid_cols)

if rows_limit:
    grid_rows = grid_rows[:rows_limit]
if cols_limit:
    grid_cols = grid_cols[:cols_limit]

y_roi = int(sizeY * (percent_overlap + 0.05))


def fetch_image(dataset, well_name, col_name, row_name):
    img = dataset[Path(well_name, col_name + row_name)].data[0, 0, 0]
    if fliplr:
        img = np.fliplr(img)
    if flipud:
        img = np.flipud(img)
    return img


# %%
row_shifts = []
for i in range(len(grid_rows) - 1):
    for col_idx, col_name in enumerate(grid_cols):
        img0 = fetch_image(dataset, well_name, col_name, grid_rows[i])
        img1 = fetch_image(dataset, well_name, col_name, grid_rows[i + 1])

        shift, _, _ = phase_cross_correlation(
            img0[-y_roi:, :], img1[:y_roi, :], upsample_factor=10
        )
        shift[0] += sizeX - y_roi
        row_shifts.append(shift)
row_translation = np.median(row_shifts, axis=0)[::-1]

col_shifts = []
for j in range(len(grid_cols) - 1):
    for row_idx, row_name in enumerate(grid_rows):
        img0 = fetch_image(dataset, well_name, grid_cols[j], row_name)
        img1 = fetch_image(dataset, well_name, grid_cols[j + 1], row_name)

        shift, _, _ = phase_cross_correlation(
            img0[:, -y_roi:], img1[:, :y_roi], upsample_factor=10
        )
        shift[1] += sizeY - y_roi
        col_shifts.append(shift)

col_translation = np.median(col_shifts, axis=0)[::-1]

# %%
print(f'Column translation: {col_translation}, row translation: {row_translation}')