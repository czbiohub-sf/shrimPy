import numpy as np

from colorspacious import cspace_convert
from skimage.color import hsv2rgb
from skimage.exposure import rescale_intensity


def HSV_PRO(czyx, channel_order, max_val_V: float = 1.0, max_val_S: float = 1.0):
    """
    HSV encoding of retardance + orientation + phase image with hsv colormap (orientation in h, retardance in s, phase in v)
    Parameters
    ----------
        czyx : numpy.ndarray
        channel_order: list
                    the order in which the channels should be stacked i.e([orientation_c_idx, retardance_c_idx,phase_c_idx])
                    the 0 index corresponds to the orientation image (range from 0 to pi)
                    the 1 index corresponds to the retardance image
                    the 2 index corresponds to the Phase image
        max_val_V   : float
                      raise the brightness of the phase channel by 1/max_val_V
        max_val_S   : float
                      raise the brightness of the retardance channel by 1/max_val_S
    Returns:
        RGB with HSV (Orientation, Retardance, Phase)
    """

    C, Z, Y, X = czyx.shape
    assert C == 3, "The input array must have 3 channels"
    print(f"channel_order: {channel_order}")

    czyx_out = np.zeros((3, Z, Y, X), dtype=np.float32)
    # Normalize the stack
    ordered_stack = np.stack(
        (
            # Normalize the first channel by dividing by pi
            czyx[channel_order[0]] / np.pi,
            # Normalize the second channel and rescale intensity
            rescale_intensity(
                czyx[channel_order[1]],
                in_range=(
                    np.min(czyx[channel_order[1]]),
                    np.max(czyx[channel_order[1]]),
                ),
                out_range=(0, 1),
            )
            / max_val_S,
            # Normalize the third channel and rescale intensity
            rescale_intensity(
                czyx[channel_order[2]],
                in_range=(
                    np.min(czyx[channel_order[2]]),
                    np.max(czyx[channel_order[2]]),
                ),
                out_range=(0, 1),
            )
            / max_val_V,
        ),
        axis=0,
    )
    czyx_out = hsv2rgb(ordered_stack, channel_axis=0)
    return czyx_out


def HSV_RO(czyx, channel_order: list[int], max_val_V: int = 1):
    """
    visualize retardance + orientation with hsv colormap (orientation in h, saturation=1 s, retardance in v)
    Parameters
    ----------
        czyx : numpy.ndarray
        channel_order: list
                    the order in which the channels should be stacked i.e([orientation_c_idx, retardance_c_idx])
                    the 0 index corresponds to the orientation image (range from 0 to pi)
                    the 1 index corresponds to the retardance image
        max_val_V   : float
                    raise the brightness of the phase channel by 1/max_val_V
    Returns:

        RGB with HSV (Orientation, _____ , Retardance)
    """
    C, Z, Y, X = czyx.shape
    assert C == 2, "The input array must have 2 channels"
    czyx_out = np.zeros((3, Z, Y, X), dtype=np.float32)
    ordered_stack = np.stack(
        (
            # Normalize the first channel by dividing by pi and then rescale intensity
            czyx[channel_order[0]] / np.pi,
            # Set the second channel to ones = Saturation 1
            np.ones_like(czyx[channel_order[0]]),
            # Normalize the third channel and rescale intensity
            np.minimum(
                1,
                rescale_intensity(
                    czyx[channel_order[1]],
                    in_range=(
                        np.min(czyx[channel_order[1]]),
                        np.max(czyx[channel_order[1]]),
                    ),
                    out_range=(0, max_val_V),
                ),
            ),
        ),
        axis=0,
    )
    # HSV-RO encoding
    czyx_out = hsv2rgb(ordered_stack, channel_axis=0)
    return czyx_out


def JCh_mapping(czyx, channel_order: list[int], max_val_ret: int = None, noise_level: int = 1):
    """
    JCh retardance + orientation + phase image with hsv colormap (orientation in h, retardance in s, phase in v)
    Parameters
    ----------
        czyx : numpy.ndarray
        channel_order: list
                     the order in which the channels should be stacked i.e([retardance_c_idx, orientation_c_idx])
                      the 0 index corresponds to the retardance image
                      the 1 index corresponds to the orientation image (range from 0 to pi)

        max_val_V   : float
                      raise the brightness of the phase channel by 1/max_val_ret
    Returns:
        RGB with JCh (Retardance, Orientation)
    """
    # retardance, orientation
    C, Z, Y, X = czyx.shape
    assert C == 2, "The input array must have 2 channels"

    # Retardance,chroma,Hue
    czyx_out = np.zeros((3, Z, Y, X), dtype=np.float32)
    for z_idx in range(Z):
        # Retardance
        if max_val_ret is None:
            max_val_ret = np.max(czyx[channel_order[0], z_idx])
        retardance = np.clip(czyx[channel_order[0], z_idx], 0, max_val_ret)
        # Chroma of each pixel, set to 60 by default, with noise handling
        chroma = np.where(czyx[channel_order[0], z_idx] < noise_level, 0, 60)
        # Orientation 180 to 360 to match periodic hue
        hue = czyx[channel_order[1], z_idx] * 360 / np.pi
        # Stack arrays in the correct order (Y, X, 3)
        I_JCh = np.stack((retardance, chroma, hue), axis=-1)
        # Transpose to shape for the skimage or colorspace functions
        JCh_rgb = cspace_convert(I_JCh, "JCh", "sRGB1")
        JCh_rgb = np.clip(JCh_rgb, 0, 1)
        czyx_out[:, z_idx] = np.transpose(JCh_rgb, (2, 0, 1))

    return czyx_out
