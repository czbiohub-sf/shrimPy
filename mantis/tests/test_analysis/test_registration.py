import numpy as np
from mantis.analysis import registration


def test_register_data():
    # Create a sample 3D volume
    raw_data = np.zeros((3, 4, 4))
    raw_data[1, 1, 1] = 1

    # Identity transformation matrix
    affine_matrix = np.eye(3)

    # Output shape
    output_shape = (3, 4, 4)

    # Test registration with identity matrix (no transformation)
    registered_data = registration.register_data(raw_data, affine_matrix, output_shape)
    np.testing.assert_array_equal(registered_data, raw_data)

    # Test inverse registration
    inverse_affine_matrix = np.linalg.inv(affine_matrix)
    inverse_registered_data = registration.register_data(
        raw_data, inverse_affine_matrix, output_shape, inverse=True
    )
    np.testing.assert_array_almost_equal(inverse_registered_data, raw_data)
