# %%
import time
from pathlib import Path
import numpy as np
import torch
from iohub import open_ome_zarr
import napari
from mantis.acquisition.autotracker import vs_inference_t2t
from viscy.translation.engine import VSUNet, AugmentedPredictionVSUNet
from mantis.acquisition.autotracker import phase_cross_corr
from skimage.registration import phase_cross_correlation
from waveorder.models.phase_thick_3d import apply_inverse_transfer_function, calculate_transfer_function
import gc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# === 1) Build model ===
config = {
    "gpu": True,
    "model": {
        "class_path": "viscy.translation.engine.VSUNet",
        "init_args": {
            "architecture": "fcmae",
            "model_config": {
                "in_channels": 1,
                "out_channels": 2,
                "in_stack_depth": 21,
                "encoder_blocks": [3, 3, 9, 3],
                "dims": [96, 192, 384, 768],
                "encoder_drop_path_rate": 0.0,
                "stem_kernel_size": [7, 4, 4],
                "in_stack_depth": 21,
                "decoder_conv_blocks": 2,
                "pretraining": False,
                "head_conv": True,
                "head_conv_expansion_ratio": 4,
                "head_conv_pool": False,
            },
        },
        "test_time_augmentations": True,
        "tta_type": "median",
    },
    "ckpt_path": "/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt"
}

phase_config ={
    "transfer_function":{
      "wavelength_illumination": 0.450,
      "yx_pixel_size": 0.1494,
      "z_pixel_size": 0.174,
      "z_padding": 5,
      "index_of_refraction_media": 1.4,
      "numerical_aperture_detection": 1.35,
      "numerical_aperture_illumination": 0.52,
      "invert_phase_contrast": False},
    "apply_inverse":{
      "reconstruction_algorithm": "Tikhonov",
      "regularization_strength": 0.01,
      "TV_rho_strength": 0.001,
      "TV_iterations": 1,}
      }

# === 2) Read input volume from OME-Zarr ===
root = Path("/hpc/projects/tlg2_mantis/")
dataset = "2025_08_01_zebrafish_golden_trio"
BF_path = root / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr/0/1/000000"


with open_ome_zarr(BF_path) as ds:
    T, C, Z, Y, X = ds.data.shape
    print("Input shape:", (T, C, Z, Y, X))
    volume_np_0 = np.asarray(ds.data[0, 0])  # (Z, Y, X)
    volume_np_10 = np.asarray(ds.data[10, 0])

vol = torch.as_tensor(volume_np_0, dtype=torch.float32, device=DEVICE)
vol_10 = torch.as_tensor(volume_np_10, dtype=torch.float32, device=DEVICE)


phase_config['transfer_function']['zyx_shape'] = (Z, Y, X)
transfer_function = tuple(tf.to(DEVICE) for tf in calculate_transfer_function(**phase_config['transfer_function']))

vol_phase = apply_inverse_transfer_function(vol, *transfer_function, **phase_config['apply_inverse'], z_padding=phase_config['transfer_function']['z_padding'])
vol_10_phase = apply_inverse_transfer_function(vol_10, *transfer_function, **phase_config['apply_inverse'], z_padding=phase_config['transfer_function']['z_padding'])




t0 = time.time()
pred = vs_inference_t2t(vol_phase.unsqueeze(0).unsqueeze(0), config)
pred_10 = vs_inference_t2t(vol_10_phase.unsqueeze(0).unsqueeze(0), config)

del vol_phase, vol_10_phase, vol, vol_10
gc.collect(); torch.cuda.empty_cache()

torch.cuda.synchronize() if DEVICE.type == "cuda" else None
print(f"Inference time: {time.time() - t0:.2f} seconds")

# === 4) Visualize in Napari ===
pred_np = pred.detach().cpu().numpy()
pred_np_10 = pred_10.detach().cpu().numpy()
del pred, pred_10
nuc = pred_np[0, 0]
mem = pred_np[0, 1]
nuc_10 = pred_np_10[0, 0]
mem_10 = pred_np_10[0, 1]

viewer = napari.Viewer()
viewer.add_image(volume_np_0, name="phase_input", colormap="gray")
viewer.add_image(volume_np_10, name="phase_input_10", colormap="gray")
viewer.add_image(nuc, name="virt_nuclei", colormap="magenta")
viewer.add_image(mem, name="virt_membrane", colormap="cyan")
viewer.add_image(nuc_10, name="virt_nuclei_10", colormap="magenta")
viewer.add_image(mem_10, name="virt_membrane_10", colormap="cyan")
napari.run()




#%%%
# 
import time
from pathlib import Path
import numpy as np
import torch
from iohub import open_ome_zarr
import napari

from viscy.translation.engine import VSUNet, AugmentedPredictionVSUNet
from mantis.acquisition.autotracker import phase_cross_corr
from skimage.registration import phase_cross_correlation


# %%

import importlib

import torch

from viscy.translation.engine import AugmentedPredictionVSUNet






#%%


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
ckpt_path = "/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt"  
vs = VSUNet(
    architecture="fcmae",
    model_config=dict(
        in_channels=1,
        out_channels=2,
        encoder_blocks=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        decoder_conv_blocks=2,
        stem_kernel_size=[7, 4, 4],
        in_stack_depth=21,
        pretraining=False,
        head_conv=True,
        head_conv_expansion_ratio=4,
        head_conv_pool=False,
    ),
    test_time_augmentations=True,
    tta_type="median",
    ckpt_path=ckpt_path,
).to(DEVICE).eval()

wrapper = AugmentedPredictionVSUNet(
    model=vs.model,
    forward_transforms=[lambda t: t],
    inverse_transforms=[lambda t: t],
).to(DEVICE).eval()

# 🔑 Must call to initialize internal padding logic
wrapper.on_predict_start()

# === 2) Read input volume from OME-Zarr ===
root = Path("/hpc/projects/tlg2_mantis/")
dataset = "2025_08_01_zebrafish_golden_trio"
BF_path = root / dataset / "1-preprocess/label-free/0-reconstruct" / f"{dataset}.zarr/0/1/000000"

with open_ome_zarr(BF_path) as ds:
    T, C, Z, Y, X = ds.data.shape
    print("Input shape:", (T, C, Z, Y, X))
    volume_np_0 = np.asarray(ds.data[0, 0])  # (Z, Y, X)
    volume_np_10 = np.asarray(ds.data[10, 0])

# === 3) Prepare tensor and run inference ===
vol = torch.as_tensor(volume_np_0, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
vol_10 = torch.as_tensor(volume_np_10, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
t0 = time.time()
with torch.no_grad():
    pred = wrapper.predict_sliding_windows(vol)
    pred_10 = wrapper.predict_sliding_windows(vol_10)
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
print(f"Inference time: {time.time() - t0:.2f} seconds")

# === 4) Visualize in Napari ===
pred_np = pred.detach().cpu().numpy()
pred_np_10 = pred_10.detach().cpu().numpy()
nuc = pred_np[0, 0]
mem = pred_np[0, 1]
nuc_10 = pred_np_10[0, 0]
mem_10 = pred_np_10[0, 1]

#%%
shifts_zyx = phase_cross_correlation(nuc, nuc_10, normalization=None)
print(shifts_zyx)

#%%
shifts_zyx_pcc = phase_cross_corr(nuc, nuc_10, normalization=False, transform=None)
print(shifts_zyx_pcc)

#%%
viewer = napari.Viewer()
viewer.add_image(volume_np_0, name="phase_input", colormap="gray")
viewer.add_image(volume_np_10, name="phase_input_10", colormap="gray")
viewer.add_image(nuc, name="virt_nuclei", colormap="magenta")
viewer.add_image(mem, name="virt_membrane", colormap="cyan")
viewer.add_image(nuc_10, name="virt_nuclei_10", colormap="magenta")
viewer.add_image(mem_10, name="virt_membrane_10", colormap="cyan")
napari.run()

# %%
