import ants
import numpy as np
import pytest

from mantis.analysis.register import (
    apply_affine_transform,
    convert_transform_to_ants,
    convert_transform_to_numpy,
    find_lir,
    find_overlapping_volume,
    get_3D_rescaling_matrix,
    get_3D_rotation_matrix,
)


def test_numpy_to_ants_transform_zyx():
    T_numpy = np.eye(4)
    T_ants = convert_transform_to_ants(T_numpy)
    assert isinstance(T_ants, ants.core.ants_transform.ANTsTransform)


def test_ants_to_numpy_transform_zyx():
    T_ants = ants.new_ants_transform(transform_type='AffineTransform')
    T_ants.set_parameters(np.eye(12))
    T_numpy = convert_transform_to_numpy(T_ants)
    assert isinstance(T_numpy, np.ndarray)
    assert T_numpy.shape == (4, 4)


def test_affine_transform():
    # Create input data
    zyx_data = np.ones((10, 10, 10))
    matrix = np.eye(4)
    output_shape_zyx = (10, 10, 10)

    # Call the function
    result = apply_affine_transform(zyx_data, matrix, output_shape_zyx)

    # Check the result
    assert isinstance(result, np.ndarray)
    assert result.shape == output_shape_zyx


def test_3d_translation():
    # Create input data
    zyx_data = np.ones((10, 10, 10))
    matrix = np.eye(4)
    translation = np.array([-3, 1, 4])
    matrix[:3, -1] = translation
    output_shape_zyx = (10, 10, 10)

    # Call the function
    result = apply_affine_transform(zyx_data, matrix, output_shape_zyx)

    # Check the result
    assert isinstance(result, np.ndarray)
    assert result.shape == output_shape_zyx
    assert np.all(
        result[3:10, 0:9, 0:6] == 1
    )  # Test if the shifts where going to the right direction


@pytest.mark.parametrize(
    "function_to_test, parameters",
    [
        (get_3D_rescaling_matrix, {"start_shape_zyx": (10, 10, 10)}),
        (get_3D_rotation_matrix, {"start_shape_zyx": (10, 10, 10)}),
        (convert_transform_to_ants, {"T_numpy": np.eye(4)}),
        (convert_transform_to_numpy, {"T_ants": convert_transform_to_ants(np.eye(4))}),
        (
            apply_affine_transform,
            {
                "zyx_data": np.random.rand(10, 10, 10),
                "matrix": np.eye(4),
                "output_shape_zyx": (10, 10, 10),
            },
        ),
        (find_lir, {"registered_zyx": np.random.rand(10, 10, 10)}),
        (
            find_overlapping_volume,
            {
                "input_zyx_shape": (10, 10, 10),
                "target_zyx_shape": (10, 10, 10),
                "transformation_matrix": np.eye(4),
            },
        ),
    ],
)
def test_error(function_to_test, parameters):
    with pytest.raises(ValueError):
        function_to_test(**parameters)
