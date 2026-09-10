

#%%
import time
from pathlib import Path
import numpy as np
from iohub import open_ome_zarr
from waveorder.models.phase_thick_3d import apply_inverse_transfer_function, calculate_transfer_function
import torch
from mantis.acquisition.autotracker import phase_cross_corr as pcc
from mantis.acquisition.autotracker import vs_inference_t2t
import gc
import napari
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

vs_config = {
    "model":{
      "class_path": "viscy.translation.engine.VSUNet",
      "init_args":{
        "architecture": "fcmae",
        "model_config":{
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
          "head_conv_pool": False},
        },
    "test_time_augmentations": True,
    "tta_type": "median"}, 
    "ckpt_path": "/hpc/projects/comp.micro/virtual_staining/models/fcmae-3d/fit_v2/pretrain_end2end/lightning_logs/finetune_VS_end2end_v1_test6_prefetch2_nopersistwork_restart_2/checkpoints/epoch=64-step=24960.ckpt"
  }


def minmax_normalize(volume: torch.Tensor) -> torch.Tensor:
    vmin = volume.min()
    vmax = volume.max()
    return (volume - vmin) / (vmax - vmin + 1e-8)

def standardize_tensor(volume: torch.Tensor) -> torch.Tensor:
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + 1e-8)

def minmax_normalize_symmetric(volume: torch.Tensor) -> torch.Tensor:
    norm = minmax_normalize(volume)
    return norm * 2 - 1


def standardize_with_iqr_clipping_cpu(volume: torch.Tensor, clip_factor: float = 1.5) -> torch.Tensor:
    """
    Normalize a large 3D tensor using IQR-based clipping and z-score standardization.
    Computes quantiles on CPU using NumPy for memory safety.
    
    Parameters
    ----------
    volume : torch.Tensor
        Input 3D tensor (Z, Y, X), must be on GPU.
    clip_factor : float
        IQR multiplier (1.5 = mild outlier clipping).
    
    Returns
    -------
    torch.Tensor
        Normalized tensor (still on GPU).
    """
    device = volume.device

    volume_np = volume.detach().cpu().numpy().flatten()

    q1 = np.quantile(volume_np, 0.25)
    q3 = np.quantile(volume_np, 0.75)
    iqr = q3 - q1
    iqr =  0.03116673231124878


    lower = q1 - clip_factor * iqr
    upper = q3 + clip_factor * iqr

    # Clip and normalize on GPU
    volume_clipped = volume.clamp(min=lower, max=upper)
    mean = volume_clipped.mean()
    std = volume_clipped.std()
    median = np.median(volume_np)
    mean = -1.5235836144711357e-05
    std = 0.06819042563438416

    print("mean, std", mean, std)
    print("q1, q3", q1, q3)
    print("iqr", iqr)
    print("median", median)
    print("lower, upper", lower, upper)
    print("volume_clipped.min(), volume_clipped.max()", volume_clipped.min(), volume_clipped.max())

    return (volume_clipped - mean) / (std + 1e-8)



root = Path("/hpc/projects/tlg2_mantis/")
dataset =  "2025_08_01_zebrafish_golden_trio"

BF_path = root / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr/0/1/000000"


with open_ome_zarr(BF_path) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    volume_t0_original = np.asarray(lf_ds.data[0,0])
    volume_t10_original = np.asarray(lf_ds.data[10,0])


  
phase_config['transfer_function']['zyx_shape'] = (Z, Y, X)

t0 = time.time()

transfer_function = tuple(tf.to(DEVICE) for tf in calculate_transfer_function(**phase_config['transfer_function']))

volume_t0_t = torch.as_tensor(volume_t0_original, device=DEVICE, dtype=torch.float32)
volume_t10_t = torch.as_tensor(volume_t10_original, device=DEVICE, dtype=torch.float32)

volume_t0_phase = apply_inverse_transfer_function(volume_t0_t, *transfer_function, **phase_config['apply_inverse'], z_padding=phase_config['transfer_function']['z_padding'])
volume_t10_phase = apply_inverse_transfer_function(volume_t10_t, *transfer_function, **phase_config['apply_inverse'], z_padding=phase_config['transfer_function']['z_padding'])

del volume_t0_t, volume_t10_t
gc.collect(); torch.cuda.empty_cache()


pred_0 = vs_inference_t2t(volume_t0_phase.unsqueeze(0).unsqueeze(0), vs_config)
pred_10 = vs_inference_t2t(volume_t10_phase.unsqueeze(0).unsqueeze(0), vs_config)


pred_0_np = pred_0.detach().cpu().numpy()
pred_10_np = pred_10.detach().cpu().numpy()

nuc = pred_0_np[0, 0]
mem = pred_0_np[0, 1]

nuc_10 = pred_10_np[0, 0]
mem_10 = pred_10_np[0, 1]

del volume_t0_phase, volume_t10_phase, pred_0, pred_10
gc.collect(); torch.cuda.empty_cache()


t1 = time.time()
print("TF calc + to device:", t1 - t0, "sec")

viewer = napari.Viewer()

viewer.add_image(nuc, name="t0_nuc", colormap="magenta")
viewer.add_image(mem, name="t0_mem", colormap="cyan")
viewer.add_image(nuc_10, name="t1_nuc", colormap="magenta")
viewer.add_image(mem_10, name="t1_mem", colormap="cyan")

napari.run()

#del volume_t0, volume_t1, transfer_function  # any big tensors you don't need


#shifts_zyx
# %%
torch.cuda.synchronize()
print(torch.cuda.memory_summary())
print("Allocated:", torch.cuda.memory_allocated() / 1e9, "GB")
print("Reserved: ", torch.cuda.memory_reserved() / 1e9, "GB")

# %%

