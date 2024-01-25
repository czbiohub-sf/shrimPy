# %%
import napari
import numpy as np
import pandas as pd
import tifffile

from napari_psf_analysis.psf_analysis.extract.BeadExtractor import BeadExtractor
from napari_psf_analysis.psf_analysis.image import Calibrated3DImage
from napari_psf_analysis.psf_analysis.psf import PSF
from scipy.ndimage import uniform_filter
from skimage.feature import peak_local_max

# %%
data_path = r'Z:\2022_12_22_LS_after_SL2\epi_beads_100nm_fl_mount_after_SL2_1\LS_beads_100nm_fl_mount_after_SL2_1_MMStack_Pos0.ome.tif'
zyx_data = tifffile.imread(data_path)
spacing = (0.250, 0.069, 0.069)  # in um

# %%
viewer = napari.Viewer()
viewer.add_image(zyx_data)

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

viewer.add_points(points, name='peaks local max', size=12, symbol='ring', edge_color='yellow')

# %%

patch_size = (spacing[0] * 10, spacing[1] * 15, spacing[2] * 15)
# round to nearest 0.5 um
patch_size = np.round(np.asarray(patch_size) / 0.5) * 0.5

# extract bead patches
bead_extractor = BeadExtractor(
    image=Calibrated3DImage(data=zyx_data.astype(np.float64), spacing=spacing),
    patch_size=patch_size,
)
beads = bead_extractor.extract_beads(points=points)
bead_offsets = np.asarray([bead.offset for bead in beads])


def analyze_psf(bead: Calibrated3DImage):
    psf = PSF(image=bead)
    try:
        psf.analyze()
        return psf.get_summary_dict()
    except RuntimeError:
        # skip over beads where psf analysis failed
        return {}


# analyze bead patches
results = [analyze_psf(bead) for bead in beads]
num_failed = sum([result == {} for result in results])

df = pd.DataFrame.from_records(results)
df['z_mu'] += bead_offsets[:, 0] * spacing[0]
df['y_mu'] += bead_offsets[:, 1] * spacing[1]
df['x_mu'] += bead_offsets[:, 2] * spacing[2]
df = df.dropna()

# %%
