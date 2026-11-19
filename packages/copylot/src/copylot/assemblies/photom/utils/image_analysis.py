"""Spot detection for photom mirror calibration.

Vendored from coPylot ``copylot/assemblies/photom/utils/image_analysis.py``
(main @ 0bcc3a3), BSD-3-Clause; see the package LICENSE.

Requires scikit-image (and matplotlib for :func:`plot_centroids`), from the
``calibration`` extra. ``plot_centroids`` uses matplotlib's ``Figure`` API rather
than pyplot (upstream raised ``NameError`` when ``mip=False`` and was not safe
off the main thread).
"""

from __future__ import annotations

import numpy as np

from skimage import filters
from skimage.feature import peak_local_max
from skimage.filters import gaussian


def find_objects_centroids(
    image, sigma=5, threshold_rel=0.5, min_distance=10, max_num_peaks=1
) -> np.ndarray:
    """Centroids of blurred objects (e.g. the laser spot) in a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D image.
    sigma : float
        Standard deviation of the Gaussian smoothing.
    threshold_rel : float
        Peak threshold, relative to the Otsu threshold of the smoothed image.
    min_distance : int
        Minimum distance between peaks (and from the image border).
    max_num_peaks : int
        Maximum number of peaks to return.

    Returns
    -------
    np.ndarray
        ``(N, 2)`` array of ``(row, column)`` centroids.
    """
    smoothed_image = gaussian(image, sigma=sigma)
    threshold_value = filters.threshold_otsu(smoothed_image)
    return peak_local_max(
        smoothed_image,
        min_distance=min_distance,
        threshold_abs=threshold_value * threshold_rel,
        num_peaks=max_num_peaks,
        exclude_border=True,
    )


def plot_centroids(image_sequence, centroids, mip=True, save_path=None):
    """Plot centroids over the image sequence (its max projection if ``mip``).

    Parameters
    ----------
    image_sequence : np.ndarray
        ``(T, Y, X)`` image sequence.
    centroids : np.ndarray
        ``(N, 2)`` array of ``(row, column)`` centroids.
    save_path : Path, optional
        Where to save the MIP figure.
    """
    # The object-oriented Figure API never touches pyplot's global state or a
    # GUI backend, so this is safe from a worker thread (upstream's pyplot
    # version was not, under Qt).
    from matplotlib.figure import Figure

    if not mip:
        # Upstream drew one figure per frame without saving or showing them.
        return

    fig = Figure()
    ax = fig.add_subplot()
    ax.imshow(np.max(image_sequence, axis=0), cmap="gray")
    for i, centroid in enumerate(centroids):
        ax.scatter(centroid[1], centroid[0], color="red")
        ax.text(centroid[1], centroid[0], str(i + 1), color="white", fontsize=8)
    if save_path is not None:
        fig.savefig(save_path)


def calculate_centroids(image_sequence, sigma=5, threshold_rel=0.5, min_distance=10):
    """:func:`find_objects_centroids` applied to each frame of a sequence."""
    return np.array(
        [
            find_objects_centroids(
                frame, sigma=sigma, threshold_rel=threshold_rel, min_distance=min_distance
            )
            for frame in image_sequence
        ]
    )
