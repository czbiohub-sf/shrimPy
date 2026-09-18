import os
import tifffile
import glob
import napari
import numpy as np
from waveorder.focus import focus_from_transverse_band

data_path = r'D:\2023_07_07_O3_autofocus'
dataset = 'kidney_rfp_fov0'

viewer = napari.Viewer()
files = glob.glob(os.path.join(data_path, dataset, '*.ome.tif'))

data = []
points = []
for i, file in enumerate(files):
    stack = tifffile.imread(file, is_ome=False)
    focus_idx = focus_from_transverse_band(stack, NA_det=1.35, lambda_ill=0.55, pixel_size=6.5/(40*1.4))
    data.append(stack)
    points.append([i, focus_idx, 50, 50])

viewer.add_image(np.asarray(data))
viewer.add_points(np.asarray(points), size=20)
napari.run()