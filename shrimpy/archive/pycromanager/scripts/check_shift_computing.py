#%%
import iohub
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from scipy.signal import correlate2d
from mantis.acquisition.autotracker import phase_cross_corr as pcc
from scipy.fft import fftn, ifftn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import cast
from scipy.fft import next_fast_len
from mantis.acquisition.autotracker import _match_shape
from iohub import open_ome_zarr

def plot_cross_correlation(
    corr,
    title="Cross-Correlation",
    output_path=None,
    xlabel="X shift (pixels)",
    ylabel="Y shift (pixels)",
) -> None:
    """
    Plot the cross-correlation.

    Parameters
    ----------
    corr : ArrayLike
        Cross-correlation array.
    title : str
        Title for the plot.
    output_path : Path
        Path to the output directory.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.

    Returns
    -------
    None
        Saves the plot to the output directory.
    """
    # Convert to 2D if necessary
    if corr.ndim == 3:
        corr_to_plot = np.max(corr, axis=0)  # Or a center slice
    else:
        corr_to_plot = corr

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_to_plot, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation strength")

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    plt.close(fig)  # This prevents overlap in future plots


def plot_spectrum(
        Fimg,
        title="FFT Spectrum",
        log_scale=True,
        cmap="gray",
        output_path=None,
        xlabel="X (pixels)",
        ylabel="Y (pixels)"):
    if Fimg.ndim == 2:
        Fimg_shifted = np.fft.fftshift(Fimg)
    else:
        Fimg_shifted = np.fft.fftshift(Fimg, axes=(-2, -1))

    magnitude = np.abs(Fimg_shifted)
    if log_scale:
        magnitude = np.log1p(magnitude)

    if magnitude.ndim == 3:
        magnitude_to_plot = np.max(magnitude, axis=0)  # or take a slice: magnitude[magnitude.shape[0]//2]
    else:
        magnitude_to_plot = magnitude

    plt.imshow(magnitude_to_plot, cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="log(1 + |FFT|)" if log_scale else "|FFT|")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()



def plot_corr(corr, title="Cross-Correlation", output_path=None,
              xlabel="X shift (pixels)", ylabel="Y shift (pixels)"):
    # Convert to 2D if necessary
    if corr.ndim == 3:
        corr_to_plot = np.max(corr, axis=0)  # or use a slice: corr[corr.shape[0] // 2]
    else:
        corr_to_plot = corr

    plt.imshow(corr_to_plot, cmap="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label="Correlation strength")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()

def _our_pcc_step_by_step_old(
    ref_img,
    mov_img,
    maximum_shift=1.0,
    normalization="classic",
    output_path=None,
    verbose=False,
):
    """
    Borrowing from Jordao dexpv2.crosscorr https://github.com/royerlab/dexpv2

    Computes translation shift using arg. maximum of phase cross correlation.
    Input are padded or cropped for fast FFT computation assuming a maximum translation shift.

    Parameters
    ----------
    ref_img : ArrayLike
        Reference image.
    mov_img : ArrayLike
        Moved image.
    maximum_shift : float, optional
        Maximum location shift normalized by axis size, by default 1.0

    Returns
    -------
    Tuple[int, ...]
        Shift between reference and moved image.
    """
    shape = tuple(
        cast(int, next_fast_len(int(max(s1, s2) * maximum_shift)))
        for s1, s2 in zip(ref_img.shape, mov_img.shape)
    )

    ref_img = _match_shape(ref_img, shape)
    mov_img = _match_shape(mov_img, shape)

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    if verbose:
        plot_spectrum(Fimg1, output_path=f"{output_path}_ref_freq.png")
        plot_spectrum(Fimg2, output_path=f"{output_path}_mov_freq.png")
    eps = np.finfo(Fimg1.dtype).eps
    del ref_img, mov_img

    prod = Fimg1 * Fimg2.conj()

    if normalization == "magnitude":
        norm = np.fmax(np.abs(prod), eps)
    elif normalization == "classic":
        norm = np.abs(Fimg1)*np.abs(Fimg2)
    else:
        norm = 1.0
    corr = np.fft.irfftn(prod / norm)
    del prod, norm
    del Fimg1, Fimg2

    corr = np.fft.fftshift(np.abs(corr))
    if verbose:
        plot_corr(corr, output_path=f"{output_path}_corr.png")
    argmax = np.argmax(corr)
    peak = np.unravel_index(argmax, corr.shape)
    peak = tuple(s // 2 - p for s, p in zip(corr.shape, peak))


    return peak, corr

def _our_pcc_step_by_step(ref_img, mov_img, normalize=None, output_path=None, verbose=False):

    Fimg1 = np.fft.rfftn(ref_img)
    Fimg2 = np.fft.rfftn(mov_img)
    if verbose:
        plot_spectrum(Fimg1, output_path=f"{output_path}_ref_freq.png")
        plot_spectrum(Fimg2, output_path=f"{output_path}_mov_freq.png")

    prod = Fimg1* Fimg2.conj() #cross-spectrum

    if normalize == "magnitude":
        eps = np.finfo(Fimg1.dtype).eps
        norm = np.fmax(np.abs(prod), eps)
    elif normalize == "classic":
        norm = np.abs(Fimg1)*np.abs(Fimg2)
    else:
        norm = 1
    corr = np.fft.irfftn(prod / norm)
    corr_shifted = np.fft.fftshift(np.abs(corr)) # center peak # shift + magnitude
    if verbose:
        plot_corr(corr_shifted, output_path=f"{output_path}_corr.png")
    
    maxima = np.unravel_index(
        np.argmax(np.abs(corr)), corr.shape
    )
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in corr.shape])

    float_dtype = prod.real.dtype

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(corr.shape)[shift > midpoint]

    return shift, corr_shifted

def skimage_pcc_step_by_step(ref_img, mov_img, normalization="phase", output_path=None, verbose=False):
    src_freq = fftn(ref_img)       
    target_freq = fftn(mov_img)
    if verbose:
        plot_spectrum(src_freq, output_path=f"{output_path}_skimage_src_freq.png")
        plot_spectrum(target_freq, output_path=f"{output_path}_skimage_target_freq.png")
    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if normalization == "phase":
        eps = np.finfo(image_product.real.dtype).eps
        image_product /= np.maximum(np.abs(image_product), 100 * eps)
    elif normalization is not None:
        raise ValueError("normalization must be either phase or None")
    cross_correlation = ifftn(image_product)
    cross_correlation_shifted = np.fft.fftshift(cross_correlation)
    if verbose:
        plot_corr(cross_correlation_shifted.real, output_path=f"{output_path}_skimage_corr.png")

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
    midpoint = np.array([np.fix(axis_size / 2) for axis_size in shape])

    float_dtype = image_product.real.dtype

    shift = np.stack(maxima).astype(float_dtype, copy=False)
    shift[shift > midpoint] -= np.array(shape)[shift > midpoint]

    return shift, cross_correlation_shifted.real


def plot_imgs_2d(img_ref, img_mov, corrected_img, output_path, title):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_ref, cmap="gray")
    axs[0].set_title("Reference Image")
    axs[1].imshow(img_mov, cmap="gray")
    axs[1].set_title("Moving Image")
    axs[2].imshow(corrected_img, cmap="gray")
    axs[2].set_title("Registered Image")
    plt.title(title)
    plt.tight_layout()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.savefig(output_path)
    plt.show()


def plot_imgs_3d(img_ref, img_mov, corrected_img, output_path, title):
    if img_ref.ndim == 3:
        img_ref_xy = img_ref[img_ref.shape[0] // 2, :, :]
        img_ref_xz = img_ref[:, img_ref.shape[1] // 2, :]
        img_ref_yz = img_ref[:, :, img_ref.shape[2] // 2]
    else:
        raise ValueError("Reference image must be 3D")

    if img_mov.ndim == 3:
        img_mov_xy = img_mov[img_mov.shape[0] // 2, :, :]
        img_mov_xz = img_mov[:, img_mov.shape[1] // 2, :]
        img_mov_yz = img_mov[:, :, img_mov.shape[2] // 2]
    else:
        raise ValueError("Moving image must be 3D")

    if corrected_img.ndim == 3:
        corrected_img_xy = corrected_img[corrected_img.shape[0] // 2, :, :]
        corrected_img_xz = corrected_img[:, corrected_img.shape[1] // 2, :]
        corrected_img_yz = corrected_img[:, :, corrected_img.shape[2] // 2]
    else:
        raise ValueError("Corrected image must be 3D")

    fig, axs = plt.subplots(3, 3, figsize=(15, 12))

    # Row 0: XY planes
    axs[0, 0].imshow(img_ref_xy, cmap="gray")
    axs[0, 0].set_title("Ref: XY")
    axs[0, 1].imshow(img_mov_xy, cmap="gray")
    axs[0, 1].set_title("Mov: XY")
    axs[0, 2].imshow(corrected_img_xy, cmap="gray")
    axs[0, 2].set_title("Reg: XY")

    # Row 1: XZ planes
    axs[1, 0].imshow(img_ref_xz, cmap="gray")
    axs[1, 0].set_title("Ref: XZ")
    axs[1, 1].imshow(img_mov_xz, cmap="gray")
    axs[1, 1].set_title("Mov: XZ")
    axs[1, 2].imshow(corrected_img_xz, cmap="gray")
    axs[1, 2].set_title("Reg: XZ")

    # Row 2: YZ planes
    axs[2, 0].imshow(img_ref_yz, cmap="gray")
    axs[2, 0].set_title("Ref: YZ")
    axs[2, 1].imshow(img_mov_yz, cmap="gray")
    axs[2, 1].set_title("Mov: YZ")
    axs[2, 2].imshow(corrected_img_yz, cmap="gray")
    axs[2, 2].set_title("Reg: YZ")

    for ax in axs.flat:
        ax.axis("off")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
    plt.savefig(output_path)
    plt.show()



def plot_imgs_zx(img_ref, img_mov, corrected_img, output_path, title):
    if img_ref.ndim == 3:
        img_ref = img_ref[:, img_ref.shape[1]//2, :]
    if img_mov.ndim == 3:
        img_mov = img_mov[:, img_mov.shape[1]//2, :]
    if corrected_img.ndim == 3:
        corrected_img = corrected_img[:, corrected_img.shape[1]//2, :]
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_ref, cmap="gray")
    axs[0].set_title("Reference Image")
    axs[1].imshow(img_mov, cmap="gray")
    axs[1].set_title("Moving Image")
    axs[2].imshow(corrected_img, cmap="gray")
    axs[2].set_title("Registered Image")
    plt.title(title)
    plt.tight_layout()
    plt.xlabel("X (pixels)")
    plt.ylabel("Z (pixels)")
    plt.savefig(output_path)
    plt.show()

def process_timepoint(ref_image, mov_image, output_path=None, verbose=False):

    shift_padding_classic, corr_padding_classic = _our_pcc_step_by_step_old(ref_image, mov_image, output_path=output_path, normalization="classic", verbose=verbose)
    shift_padding_magnitude, corr_padding_magnitude = _our_pcc_step_by_step_old(ref_image, mov_image, output_path=output_path, normalization="magnitude", verbose=verbose)
    shift_classic, corr_classic = _our_pcc_step_by_step(ref_image, mov_image, normalize="classic", output_path=output_path, verbose=verbose)
    shift_magnitude, corr_magnitude = _our_pcc_step_by_step(ref_image, mov_image, normalize="magnitude", output_path=output_path, verbose=verbose)
    shift_skimage, corr_skimage = skimage_pcc_step_by_step(ref_image, mov_image, output_path=output_path, verbose=verbose, normalization="phase")
    return shift_padding_classic, corr_padding_classic, shift_padding_magnitude, corr_padding_magnitude, shift_classic, corr_classic, shift_magnitude, corr_magnitude, shift_skimage, corr_skimage


def process_channel(image_path, position_keys, output_dir=None, verbose=False):
    dict_shifts_padding_classic = {}
    dict_shifts_padding_magnitude = {}
    dict_shifts_norm_magnitude = {}
    dict_shifts_norm_classic = {}
    dict_shifts_skimage = {}
    print("processing channel:", image_path)

    for position_key in tqdm(position_keys):
        print("position_key:", position_key)
        with open_ome_zarr(image_path /Path(*position_key)) as bf_ds:
            T, C,Z, Y, X = bf_ds.data.shape
            scale = bf_ds.scale
            print("scale:", scale)
            print("T:{} C:{} Z:{} Y:{} X:{}".format(T, C, Z, Y, X))
            ref_image = np.asarray(bf_ds.data[0,0,:,:,:])
            print("reference image timepoint:", 0)
            for t in tqdm(range(1,50)):
                print("moving image timepoint:", t)
                mov_image = np.asarray(bf_ds.data[t,0,:,:,:])
                shift_padding_classic, corr_padding_classic, shift_padding_magnitude, corr_padding_magnitude, shift_classic, corr_classic, shift_magnitude, corr_magnitude, shift_skimage, corr_skimage  = process_timepoint(ref_image, mov_image, output_path=None, verbose=verbose)
                print("shift_padding_classic:", shift_padding_classic, "shift_padding_magnitude:", shift_padding_magnitude, "shift_classic:", shift_classic, "shift_magnitude:", shift_magnitude, "shift_skimage:", shift_skimage)
                # print("corr_magnitude_max:", corr_magnitude.max(), "corr_classic_max:", corr_classic.max(), "corr_skimage_max:", corr_skimage.max())
                # print("corr_magnitude_min:", corr_magnitude.min(), "corr_classic_min:", corr_classic.min(), "corr_skimage_min:", corr_skimage.min())
                # print("corr_magnitude_sum:", corr_magnitude.sum(), "corr_classic_sum:", corr_classic.sum(), "corr_skimage_sum:", corr_skimage.sum())

                position_key_str = "/".join(position_key)
                dict_shifts_norm_magnitude[position_key_str] = {"t": t, "shift": shift_magnitude, "max correlation": corr_magnitude.max(),"min correlation": corr_magnitude.min(), "correlation sum": corr_magnitude.sum()}

                dict_shifts_norm_classic[position_key_str] = {"t": t, "shift": shift_classic, "max correlation": corr_classic.max(),"min correlation": corr_classic.min(), "correlation sum": corr_classic.sum()}
                
                dict_shifts_skimage[position_key_str] = {"t": t, "shift": shift_skimage, "max correlation": corr_skimage.max(),"min correlation": corr_skimage.min(), "correlation sum": corr_skimage.sum()} 

                dict_shifts_padding_classic[position_key_str] = {"t": t, "shift": shift_padding_classic, "max correlation": corr_padding_classic.max(),"min correlation": corr_padding_classic.min(), "correlation sum": corr_padding_classic.sum()}

                dict_shifts_padding_magnitude[position_key_str] = {"t": t, "shift": shift_padding_magnitude, "max correlation": corr_padding_magnitude.max(),"min correlation": corr_padding_magnitude.min(), "correlation sum": corr_padding_magnitude.sum()}


    df_shifts_norm_magnitude = pd.DataFrame(dict_shifts_norm_magnitude)
    df_shifts_norm_classic = pd.DataFrame(dict_shifts_norm_classic)
    df_shifts_skimage = pd.DataFrame(dict_shifts_skimage)
    df_shifts_padding_classic = pd.DataFrame(dict_shifts_padding_classic)
    df_shifts_padding_magnitude = pd.DataFrame(dict_shifts_padding_magnitude)

    df_shifts_padding_classic.to_csv(output_dir / "shift_padding_classic.csv", index=False)
    df_shifts_padding_magnitude.to_csv(output_dir / "shift_padding_magnitude.csv", index=False)
    df_shifts_norm_magnitude.to_csv(output_dir / "shift_norm_magnitude.csv", index=False)
    df_shifts_norm_classic.to_csv(output_dir / "shift_norm_classic.csv", index=False)
    df_shifts_skimage.to_csv(output_dir / "shift_skimage_norm.csv", index=False)

#%%
import os
from glob import glob
from tqdm import tqdm
output_dir = Path("/hpc/projects/intracellular_dashboard/autotracker/PCC/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test/t0_ref/no_crop")
output_dir.mkdir(parents=True, exist_ok=True)
root = Path("/hpc/projects/intracellular_dashboard/autotracker")
datasets = ["2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test", "2025_05_09_52hpf_RT_zebrafish_golden_trio"]
BF_crop_slices = [(20, 90), (275, 525), (350, 680)]
LF_crop_slices = [(20, 90), (275, 525), (350, 680)]
LS_crop_slices = [(42, 117), (250, 800), (300, 710)]
dataset = datasets[0]
fov = "*/*/*"

BF_path = root / dataset / "0-convert" / f"{dataset}_symlink" / f"{dataset}_labelfree_1.zarr"
LF_path = root / dataset / "1-preprocess" / "label-free" / "0-reconstruct" / f"{dataset}.zarr"
LS_path = root / dataset / "1-preprocess" / "light-sheet" / "0-deskew" / f"{dataset}.zarr"
# get */*/*
#%%
BF_position_dirpaths = [Path(p) for p in glob(str(BF_path / fov))]
position_keys = [p.parts[-3:] for p in BF_position_dirpaths]
# sort position_keys by first element of each tuple
position_keys.sort(key=lambda x: x[0])



#%%




#%%


# with open_ome_zarr(BF_path) as bf_ds:
#     T, C,Z, Y, X = bf_ds.data.shape
#     scale = bf_ds.scale
#     print(scale)
#     print(T, C,Z, Y, X)
#     t_0 = np.asarray(bf_ds.data[0,0,:,:,:]) # take second timepoint
#     t_10 = np.asarray(bf_ds.data[10,0,:,:,:]) # take second timepoint
#     print(t_0.shape, t_10.shape)
    





# %%
path_dataset_1 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test/0-convert/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test_symlink/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test_labelfree_1.zarr/B/1/000001")

# output_path = output_dir / "exp_1"
# output_path.mkdir(parents=True, exist_ok=True)
# output_path = f"{output_path}/T0_T10_no_crop_2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test"



print("Loading data...")
with open_ome_zarr(path_dataset_1) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,:,:,:]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,:,:,:]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10

plot_imgs_3d(img_ref, img_mov, img_mov, output_path, "T0_T10")

# %%
path_dataset_2 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_05_09_52hpf_RT_zebrafish_golden_trio/0-convert/2025_05_09_52hpf_RT_zebrafish_golden_trio_symlink/2025_05_09_52hpf_RT_zebrafish_golden_trio_labelfree_1.zarr/B/1/000")
output_path = output_dir / "exp_3"
output_path.mkdir(parents=True, exist_ok=True)
output_path = f"{output_path}/T0_T10_no_crop_2025_05_09_52hpf_RT_zebrafish_golden_trio"



print("Loading data...")
with open_ome_zarr(path_dataset_2) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,:,:,:]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,:,:,:]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10



# %%
path_dataset_3 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_05_09_52hpf_RT_zebrafish_golden_trio/1-preprocess/light-sheet/raw/0-deskew/2025_05_09_52hpf_RT_zebrafish_golden_trio.zarr/B/1/000")
output_path = output_dir / "exp_4"
output_path.mkdir(parents=True, exist_ok=True)
output_path = f"{output_path}/T0_T10_no_crop_2025_05_09_52hpf_RT_zebrafish_golden_trio_lightsheet"



print("Loading data...")
with open_ome_zarr(path_dataset_3) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,:,:,:]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,:,:,:]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10


# %% CROPPING
path_dataset_1 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test/0-convert/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test_symlink/2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test_labelfree_1.zarr/B/1/000004")
output_path = output_dir / "cropping" / "exp_1"
output_path.mkdir(parents=True, exist_ok=True)
output_path = f"{output_path}/T0_T10_crop_2025_03_27_zebrafish_myo6b_2dpf_bf_autotracker_test"


with open_ome_zarr(path_dataset_1) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,20:90,275:525,350:680]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,10:80,275:525,350:680]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10
plot_imgs_3d(img_ref, img_mov, img_mov, output_path, "T0_T10")

# %%
path_dataset_2 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_05_09_52hpf_RT_zebrafish_golden_trio/0-convert/2025_05_09_52hpf_RT_zebrafish_golden_trio_symlink/2025_05_09_52hpf_RT_zebrafish_golden_trio_labelfree_1.zarr/B/1/000")
output_path = output_dir / "cropping" / "exp_2"
output_path.mkdir(parents=True, exist_ok=True)
output_path = f"{output_path}/T0_T10_crop_2025_05_09_52hpf_RT_zebrafish_golden_trio"


print("Loading data...")
with open_ome_zarr(path_dataset_2) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,20:90,275:525,350:680]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,10:80,275:525,350:680]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10

plot_imgs_3d(img_ref, img_mov, img_mov, output_path, "T0_T10")


# %%
path_dataset_3 = Path("/hpc/projects/intracellular_dashboard/autotracker/2025_05_09_52hpf_RT_zebrafish_golden_trio/1-preprocess/light-sheet/raw/0-deskew/2025_05_09_52hpf_RT_zebrafish_golden_trio.zarr/B/1/003")
output_path = output_dir / "cropping" / "exp_3"
output_path.mkdir(parents=True, exist_ok=True)
output_path = f"{output_path}/T0_T10_no_crop_2025_05_09_52hpf_RT_zebrafish_golden_trio_lightsheet"



print("Loading data...")
with open_ome_zarr(path_dataset_3) as lf_ds:
    T, C,Z, Y, X = lf_ds.data.shape
    scale = lf_ds.scale
    print(scale)
    print(T, C,Z, Y, X)
    t_0 = np.asarray(lf_ds.data[0,0,42:117,250:800,300:710]) # take second timepoint
    t_10 = np.asarray(lf_ds.data[10,0,42:117,250:800,300:710]) # take second timepoint
    print(t_0.shape, t_10.shape)

img_ref = t_0
img_mov = t_10

plot_imgs_3d(img_ref, img_mov, img_mov, output_path, "T0_T10")


# %%
