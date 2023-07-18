import numpy as np
import pytest
from mantis.analysis import registration

def test_estimate_transformation_matrix():
    # Test case 1: scaling
    source_points = np.array([[0, 0], [1, 0], [0, 1]])
    target_points = np.array([[0, 0], [2, 0], [0, 2]])
    expected_matrix = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    result_matrix = registration.estimate_transformation_matrix(source_points, target_points)
    np.testing.assert_allclose(result_matrix, expected_matrix, rtol=1e-3, atol=1e-3)

    # Test case 2: translation
    source_points = np.array([[0, 0], [1, 0], [0, 1]])
    target_points = np.array([[2, 2], [4, 2], [2, 4]])
    expected_matrix = np.array([[2., 0., 2.], [0., 2., 2.], [0., 0., 1.]])
    result_matrix = registration.estimate_transformation_matrix(source_points, target_points)
    np.testing.assert_allclose(result_matrix, expected_matrix, rtol=1e-3, atol=1e-3)

    # Test case 2: Shear Y
    source_points = np.array([[0, 0], [1, 0], [0, 1]])
    target_points = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    expected_matrix = np.array([[0.5, 0., 0.], [0.5, 1., 0], [0., 0., 1.]])
    result_matrix = registration.estimate_transformation_matrix(source_points, target_points)
    np.testing.assert_allclose(result_matrix, expected_matrix, rtol=1e-3, atol=1e-3)

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
    inverse_registered_data = registration.register_data(raw_data, inverse_affine_matrix, output_shape, inverse=True)
    np.testing.assert_array_equal(inverse_registered_data, raw_data)
