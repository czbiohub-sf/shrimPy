import numpy as np
import scipy


def register_data(
    raw_data: np.ndarray,
    affine_matrix: np.ndarray,
    output_shape: tuple,
    inverse=False,
    order: int = 1,
    cval: float = None,
):
    """
    Apply the affine transform on the zyx volume

    Parameters
    ----------
    raw_data : np.ndarray with ndim==3 (z,y,x)
        volume to be transformed in (z,y,x) order
    affine_transform : _type_
        _description_
    output_shape: Tuple
        tuple with the output shape of output
    order : int, optional
        _description_, by default 1
    cval : float, optional
        _description_, by default None
    """
    if cval is None:
        cval = np.min(np.ravel(raw_data))

    Z, Y, X = raw_data.shape

    if inverse:
        affine_matrix = np.linalg.inv(affine_matrix)

    registered_data = scipy.ndimage.affine_transform(
        raw_data, affine_matrix, output_shape=output_shape, order=order, cval=cval
    )

    return registered_data
