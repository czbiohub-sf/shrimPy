import numpy as np
import scipy

def estimate_transformation_matrix(source_points, target_points):
    # Construct the matrix equation
    A = np.zeros((len(source_points) * 2, 6))
    b = np.zeros((len(source_points) * 2, 1))

    for i in range(len(source_points)):
        x, y = source_points[i]
        u, v = target_points[i]

        A[i*2, :] = [x, y, 1, 0, 0, 0]
        A[i*2+1, :] = [0, 0, 0, x, y, 1]
        b[i*2, 0] = u
        b[i*2+1, 0] = v

    # Solve the matrix equation using least squares
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Extract the transformation matrix values
    a, b, c, d, e, f = x.flatten()

    # Construct the transformation matrix
    transformation_matrix = np.array([[a, b, c],
                                      [d, e, f],
                                      [0, 0, 1]])

    return transformation_matrix

def register_data(raw_data: np.ndarray,
                    affine_matrix:np.ndarray,
                    output_shape:tuple,
                    inverse=False,
                    order: int = 1,
                    cval:float=None
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

    Z,Y,X = raw_data.shape

    if inverse:
        affine_matrix = np.linalg.inv(affine_matrix)

    registered_data = scipy.ndimage.affine_transform(
        raw_data,
        affine_matrix,
        output_shape= output_shape,
        order=order,
        cval=cval
    )

    return registered_data


