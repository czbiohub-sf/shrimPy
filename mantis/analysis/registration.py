import numpy as np


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
