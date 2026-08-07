# %%
import time
from pathlib import Path
import numpy as np
import torch
from iohub import open_ome_zarr
import napari
import importlib

from viscy.translation.engine import VSUNet, AugmentedPredictionVSUNet
from mantis.acquisition.autotracker import phase_cross_corr
from skimage.registration import phase_cross_correlation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def vs_inference_t2t(x: torch.Tensor, cfg: dict, gpu: bool = True) -> torch.Tensor:
    """
    Run virtual staining using a config dictionary and 5D input tensor (B, C, Z, Y, X).
    Returns predicted tensor of shape (B, C_out, Z, Y, X).
    """
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Extract model info
    model_cfg = cfg["model"].copy()
    init_args = model_cfg["init_args"]
    class_path = model_cfg["class_path"]

    # Inject ckpt_path from top-level config if needed
    if "ckpt_path" in cfg:
        init_args["ckpt_path"] = cfg["ckpt_path"]

    # Import model class dynamically
    module_path, class_name = class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)

    # Instantiate model
    model = model_class(**init_args).to(device).eval()

    # Wrap with augmentation logic
    wrapper = (
        AugmentedPredictionVSUNet(
            model=model.model,
            forward_transforms=[lambda t: t],
            inverse_transforms=[lambda t: t],
        )
        .to(x.device)
        .eval()
    )

    wrapper.on_predict_start()
    return wrapper.predict_sliding_windows(x)


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

# === 2) Read input volume from OME-Zarr ===
root = Path("/hpc/projects/tlg2_mantis/")
dataset = "2025_08_01_zebrafish_golden_trio"
BF_path = root / dataset / "1-preprocess/label-free/0-reconstruct" / f"{dataset}.zarr/0/1/000000"


with open_ome_zarr(BF_path) as ds:
    T, C, Z, Y, X = ds.data.shape
    print("Input shape:", (T, C, Z, Y, X))
    volume_np_0 = np.asarray(ds.data[0, 0])  # (Z, Y, X)
    volume_np_10 = np.asarray(ds.data[10, 0])

vol = torch.as_tensor(volume_np_0, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
vol_10 = torch.as_tensor(volume_np_10, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
t0 = time.time()
pred = vs_inference_t2t(vol, config)
pred_10 = vs_inference_t2t(vol_10, config)
torch.cuda.synchronize() if DEVICE.type == "cuda" else None
print(f"Inference time: {time.time() - t0:.2f} seconds")

# === 4) Visualize in Napari ===
pred_np = pred.detach().cpu().numpy()
pred_np_10 = pred_10.detach().cpu().numpy()
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
