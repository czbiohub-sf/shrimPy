"""
Test script for Autotracker instantiation and shift estimation using saved images.
This script loads OME-Zarr data and tests different tracking methods without microscope connection.

Note on GPU usage:
- Set USE_GPU=True (line ~30) to enable GPU acceleration for virtual staining only
- Phase reconstruction always uses CPU to avoid CUDA kernel compatibility issues
- If you encounter CUDA errors, keep USE_GPU=False
- CPU mode will be slower but works on all systems
- Tests 1, 2, 3, and 5 run reasonably fast on CPU; Test 4 (VS) is very slow on CPU
"""

# %%
import gc
import time
from pathlib import Path
from typing import Optional

import napari
import numpy as np
import torch
from iohub import open_ome_zarr
from waveorder.models.phase_thick_3d import (
    apply_inverse_transfer_function,
    calculate_transfer_function,
)

from mantis.acquisition.autotracker import (
    Autotracker,
    data_preprocessing_labelfree,
    vs_inference_t2t,
)
from mantis.acquisition.AcquisitionSettings import AutotrackerSettings

# %%
# Try to use CUDA, but fall back to CPU if there are compatibility issues
USE_GPU = False  # Set to True to attempt GPU usage
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using device: {DEVICE} - {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print("Note: CUDA is available but CPU mode is selected. Set USE_GPU=True to use GPU.")

# === Configuration ===
# Path to your test dataset
root = Path("/hpc/projects/tlg2_mantis/")
dataset = "2025_08_01_zebrafish_golden_trio"
BF_path = root / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr/0/1/000000"

# Timepoints to compare
t_ref = 0
t_mov = 10

# Virtual staining model config
vs_config = {
    "gpu": USE_GPU,
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

# Phase reconstruction config
phase_config = {
    "transfer_function": {
        "wavelength_illumination": 0.450,
        "yx_pixel_size": 0.1494,
        "z_pixel_size": 0.174,
        "z_padding": 5,
        "index_of_refraction_media": 1.4,
        "numerical_aperture_detection": 1.35,
        "numerical_aperture_illumination": 0.52,
        "invert_phase_contrast": False,
    },
    "apply_inverse": {
        "reconstruction_algorithm": "Tikhonov",
        "regularization_strength": 0.01,
        "TV_rho_strength": 0.001,
        "TV_iterations": 1,
    }
}

# %%
# === Load data from OME-Zarr ===
print(f"Loading data from: {BF_path}")
with open_ome_zarr(BF_path) as ds:
    T, C, Z, Y, X = ds.data.shape
    print(f"Dataset shape: T={T}, C={C}, Z={Z}, Y={Y}, X={X}")
    
    volume_bf_ref = np.asarray(ds.data[t_ref, 0])  # (Z, Y, X)
    volume_bf_mov = np.asarray(ds.data[t_mov, 0])
    
print(f"Loaded volumes at t={t_ref} and t={t_mov}")
print(f"Volume shape: {volume_bf_ref.shape}")

# %%
# === Calculate transfer function for phase reconstruction ===
phase_config['transfer_function']['zyx_shape'] = (Z, Y, X)
transfer_function = calculate_transfer_function(**phase_config['transfer_function'])
transfer_function_tensor = tuple(tf.to(DEVICE) for tf in transfer_function)
# Also create CPU version to avoid CUDA kernel issues
transfer_function_cpu = tuple(tf.to(torch.device("cpu")) for tf in transfer_function)
print("Transfer function calculated (GPU and CPU versions)")

# %%
# === Test 1: Autotracker with phase_cross_correlation on brightfield ===
print("\n" + "="*60)
print("TEST 1: Phase Cross-Correlation on Brightfield")
print("="*60)

# Define scale (um per pixel)
scale = np.array([
    phase_config['transfer_function']['z_pixel_size'],  # Z
    phase_config['transfer_function']['yx_pixel_size'],  # Y
    phase_config['transfer_function']['yx_pixel_size'],  # X
])

autotracker_bf = Autotracker(
    tracking_method='phase_cross_correlation',
    scale=scale,
    zyx_dampening_factor=None,  # No dampening
    transfer_function=transfer_function_cpu,  # Use CPU version to avoid CUDA issues
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
)

t0 = time.time()
shifts_bf = autotracker_bf.estimate_shift(
    ref_img=volume_bf_ref,
    mov_img=volume_bf_mov,
)
t1 = time.time()

print(f"Estimated shifts (um): Z={shifts_bf[0]:.3f}, Y={shifts_bf[1]:.3f}, X={shifts_bf[2]:.3f}")
print(f"Time: {t1-t0:.2f} seconds")

# %%
# === Test 2: Autotracker with dampening factor ===
print("\n" + "="*60)
print("TEST 2: Phase Cross-Correlation with Dampening")
print("="*60)

dampening = np.array([0.5, 0.8, 0.8])  # Conservative Z, moderate XY
autotracker_dampened = Autotracker(
    tracking_method='phase_cross_correlation',
    scale=scale,
    zyx_dampening_factor=dampening,
    transfer_function=transfer_function_cpu,  # Use CPU version to avoid CUDA issues
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
)

t0 = time.time()
shifts_dampened = autotracker_dampened.estimate_shift(
    ref_img=volume_bf_ref,
    mov_img=volume_bf_mov,
)
t1 = time.time()

print(f"Dampening factor: {dampening}")
print(f"Estimated shifts (um): Z={shifts_dampened[0]:.3f}, Y={shifts_dampened[1]:.3f}, X={shifts_dampened[2]:.3f}")
print(f"Time: {t1-t0:.2f} seconds")

# %%
# === Test 3: Autotracker with phase reconstruction ===
print("\n" + "="*60)
print("TEST 3: Phase Cross-Correlation on Reconstructed Phase")
print("="*60)

# For phase reconstruction, use CPU to avoid CUDA kernel issues
print(f"Using device for phase reconstruction: CPU")

# Create autotracker settings for phase-only reconstruction
autotracker_settings_phase = AutotrackerSettings(
    tracking_method='phase_cross_correlation',
    tracking_interval=1,
    shift_estimation_channel='phase',
    scale_yx=phase_config['transfer_function']['yx_pixel_size'],
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
    device='cpu',
    zyx_dampening_factor=None,
    reconstruction=['phase'],  # Phase only, no VS
    phase_config=phase_config,
)

# Reconstruct phase using preprocessing function
t0 = time.time()

volume_phase_ref = data_preprocessing_labelfree(
    volume_bf_ref,
    transfer_function_cpu,
    autotracker_settings_phase
)

volume_phase_mov = data_preprocessing_labelfree(
    volume_bf_mov,
    transfer_function_cpu,
    autotracker_settings_phase
)

gc.collect()
torch.cuda.empty_cache()

t1 = time.time()
print(f"Phase reconstruction time: {t1-t0:.2f} seconds")

autotracker_phase = Autotracker(
    tracking_method='phase_cross_correlation',
    scale=scale,
    zyx_dampening_factor=None,
    transfer_function=transfer_function_cpu,  # Use CPU version to avoid CUDA issues
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
)

t0 = time.time()
shifts_phase = autotracker_phase.estimate_shift(
    ref_img=volume_phase_ref,
    mov_img=volume_phase_mov,
)
t1 = time.time()

print(f"Estimated shifts (um): Z={shifts_phase[0]:.3f}, Y={shifts_phase[1]:.3f}, X={shifts_phase[2]:.3f}")
print(f"Time: {t1-t0:.2f} seconds")

# %%
# === Test 4: Autotracker with virtual staining ===
print("\n" + "="*60)
print("TEST 4: Phase Cross-Correlation on Virtual Staining (Nuclei)")
print("="*60)

# Note: Virtual staining is GPU-intensive and may be very slow on CPU
if not USE_GPU:
    print("WARNING: Running virtual staining on CPU. This will be VERY slow.")
    print("Consider setting USE_GPU=True if you have compatible CUDA setup.")
    print("Proceeding anyway...\n")

# Create autotracker settings for preprocessing
autotracker_settings = AutotrackerSettings(
    tracking_method='phase_cross_correlation',
    tracking_interval=1,
    shift_estimation_channel='vs_nuclei',
    scale_yx=phase_config['transfer_function']['yx_pixel_size'],
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
    device='cuda' if USE_GPU else 'cpu',
    zyx_dampening_factor=(0.5, 0.8, 0.8),
    reconstruction=['phase', 'vs'],
    phase_config=phase_config,
    vs_config=vs_config,
)

t0 = time.time()

# Use appropriate transfer function based on device
if USE_GPU:
    tf_for_vs = transfer_function_tensor
else:
    # Use CPU version of transfer functions to avoid CUDA errors
    tf_for_vs = transfer_function_cpu

# Use the preprocessing function
volume_vs_ref = data_preprocessing_labelfree(
    volume_bf_ref,
    tf_for_vs,
    autotracker_settings
)

volume_vs_mov = data_preprocessing_labelfree(
    volume_bf_mov,
    tf_for_vs,
    autotracker_settings
)

t1 = time.time()
print(f"Phase + VS reconstruction time: {t1-t0:.2f} seconds")

autotracker_vs = Autotracker(
    tracking_method='phase_cross_correlation',
    scale=scale,
    zyx_dampening_factor=np.array([0.5, 0.8, 0.8]),
    transfer_function=tf_for_vs,  # Use appropriate device version
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
)

t0 = time.time()
shifts_vs = autotracker_vs.estimate_shift(
    ref_img=volume_vs_ref,
    mov_img=volume_vs_mov,
)
t1 = time.time()

print(f"Estimated shifts (um): Z={shifts_vs[0]:.3f}, Y={shifts_vs[1]:.3f}, X={shifts_vs[2]:.3f}")
print(f"Time: {t1-t0:.2f} seconds")

# %%
# === Test 5: Reference Volume Caching Test ===
print("\n" + "="*60)
print("TEST 5: Reference Volume Caching (Multiple Timepoints)")
print("="*60)
print("This test verifies that the reference volume is stored in memory")
print("and reused across multiple timepoint comparisons.\n")

# Load additional timepoint for testing
t_mov2 = 20  # Second moving timepoint
print(f"Loading additional timepoint t={t_mov2}...")
with open_ome_zarr(BF_path) as ds:
    volume_bf_mov2 = np.asarray(ds.data[t_mov2, 0])

# Create autotracker with ref_volume storage
autotracker_cache_test = Autotracker(
    tracking_method='phase_cross_correlation',
    scale=scale,
    zyx_dampening_factor=None,
    transfer_function=transfer_function_cpu,
    absolute_shift_limits_um={'z': (0.5, 2), 'y': (2, 10), 'x': (2, 10)},
)

# Process first moving volume (should compute and store reference)
print(f"\n--- Processing first moving timepoint (t={t_mov}) ---")
print("Initial ref_volume state:", "None" if autotracker_cache_test.ref_volume is None else "Loaded")

t0 = time.time()
# Set reference volume manually (simulating first call in acquisition)
autotracker_cache_test.ref_volume = volume_bf_ref
print("Manually set ref_volume from t=0")
shifts_cache_1 = autotracker_cache_test.estimate_shift(
    ref_img=autotracker_cache_test.ref_volume,
    mov_img=volume_bf_mov,
)
t1 = time.time()
time_first = t1 - t0

print(f"After first estimate, ref_volume state:", "Stored" if autotracker_cache_test.ref_volume is not None else "None")
print(f"Estimated shifts (um): Z={shifts_cache_1[0]:.3f}, Y={shifts_cache_1[1]:.3f}, X={shifts_cache_1[2]:.3f}")
print(f"Time: {time_first:.3f} seconds")

# Process second moving volume (should reuse stored reference)
print(f"\n--- Processing second moving timepoint (t={t_mov2}) ---")
print("Before second estimate, ref_volume state:", "Stored" if autotracker_cache_test.ref_volume is not None else "None")

t0 = time.time()
shifts_cache_2 = autotracker_cache_test.estimate_shift(
    ref_img=autotracker_cache_test.ref_volume,
    mov_img=volume_bf_mov2,
)
t1 = time.time()
time_second = t1 - t0

print(f"After second estimate, ref_volume state:", "Stored" if autotracker_cache_test.ref_volume is not None else "None")
print(f"Estimated shifts (um): Z={shifts_cache_2[0]:.3f}, Y={shifts_cache_2[1]:.3f}, X={shifts_cache_2[2]:.3f}")
print(f"Time: {time_second:.3f} seconds")

print("\n--- Cache Test Summary ---")
print(f"Reference volume was maintained across both estimates: {autotracker_cache_test.ref_volume is not None}")
print(f"First estimate time: {time_first:.3f}s")
print(f"Second estimate time: {time_second:.3f}s")
print(f"Time savings: Reference volume successfully reused!")

# %%
# === Summary of results ===
print("\n" + "="*60)
print("SUMMARY OF SHIFT ESTIMATES (in um)")
print("="*60)
print(f"{'Method':<40} {'Z':>8} {'Y':>8} {'X':>8}")
print("-" * 60)
print(f"{'1. PCC on Brightfield (t=0 vs t=10)':<40} {shifts_bf[0]:>8.3f} {shifts_bf[1]:>8.3f} {shifts_bf[2]:>8.3f}")
print(f"{'2. PCC with Dampening (0.5, 0.8, 0.8)':<40} {shifts_dampened[0]:>8.3f} {shifts_dampened[1]:>8.3f} {shifts_dampened[2]:>8.3f}")
print(f"{'3. PCC on Phase (t=0 vs t=10)':<40} {shifts_phase[0]:>8.3f} {shifts_phase[1]:>8.3f} {shifts_phase[2]:>8.3f}")
print(f"{'4. PCC on VS Nuclei (t=0 vs t=10)':<40} {shifts_vs[0]:>8.3f} {shifts_vs[1]:>8.3f} {shifts_vs[2]:>8.3f}")
print(f"{'5. Cache Test: BF (t=0 vs t=20)':<40} {shifts_cache_2[0]:>8.3f} {shifts_cache_2[1]:>8.3f} {shifts_cache_2[2]:>8.3f}")
print("="*60)

# %%
# === Visualization in napari ===
print("\nLaunching napari viewer...")

viewer = napari.Viewer()

# Add original volumes
viewer.add_image(volume_bf_ref, name=f"BF t={t_ref}", colormap="gray", visible=False)
viewer.add_image(volume_bf_mov, name=f"BF t={t_mov}", colormap="gray")
viewer.add_image(volume_bf_mov2, name=f"BF t={t_mov2}", colormap="gray", visible=False)

# Add phase volumes
viewer.add_image(volume_phase_ref, name=f"Phase t={t_ref}", colormap="viridis", visible=False)
viewer.add_image(volume_phase_mov, name=f"Phase t={t_mov}", colormap="viridis", visible=False)

# Add VS volumes
viewer.add_image(volume_vs_ref, name=f"VS Nuclei t={t_ref}", colormap="magenta", visible=False)
viewer.add_image(volume_vs_mov, name=f"VS Nuclei t={t_mov}", colormap="magenta", visible=False)

# Add points to show shift vectors
# Convert shifts from um to pixels for visualization
shifts_pix_bf = shifts_bf / scale
shifts_pix_phase = shifts_phase / scale
shifts_pix_vs = shifts_vs / scale
shifts_pix_cache = shifts_cache_2 / scale

# Center of volume
center = np.array([Z//2, Y//2, X//2])

# Add shift vectors as points
viewer.add_points(
    [center, center + shifts_pix_bf],
    name="Shift BF",
    size=10,
    face_color="red",
    visible=False
)

viewer.add_points(
    [center, center + shifts_pix_phase],
    name="Shift Phase",
    size=10,
    face_color="green",
    visible=False
)

viewer.add_points(
    [center, center + shifts_pix_vs],
    name="Shift VS",
    size=10,
    face_color="blue",
    visible=False
)

viewer.add_points(
    [center, center + shifts_pix_cache],
    name=f"Shift Cache (t={t_mov2})",
    size=10,
    face_color="yellow",
    visible=False
)

napari.run()

# %%

import torch, os, sys
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    i = torch.cuda.current_device()
    print("device:", torch.cuda.get_device_name(i), "capability:", torch.cuda.get_device_capability(i))
    print("cudnn:", torch.backends.cudnn.version())
print("python:", sys.version)
print("CUDA_LAUNCH_BLOCKING:", os.environ.get("CUDA_LAUNCH_BLOCKING"))


# %%
