from pathlib import Path

import numpy as np
import yaml

from iohub.ngff import open_ome_zarr
from tqdm import tqdm


def copy_n_paste(zyx_data: np.ndarray, zyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices()

    Parameters
    ----------
    zyx_data : np.ndarray
        data to copy
    zyx_slicing_params : list
        list of slicing parameters for z,y,x

    Returns
    -------
    np.ndarray
        crop of the input zyx_data given the slicing parameters
    """
    zyx_data = np.nan_to_num(zyx_data, nan=0)
    zyx_data_sliced = zyx_data[
        zyx_slicing_params[0],
        zyx_slicing_params[1],
        zyx_slicing_params[2],
    ]
    return zyx_data_sliced


def copy_n_paste_czyx(czyx_data: np.ndarray, czyx_slicing_params: list) -> np.ndarray:
    """
    Load a zyx array and crop given a list of ZYX slices()

    Parameters
    ----------
    czyx_data : np.ndarray
        data to copy
    czyx_slicing_params : list
        list of slicing parameters for z,y,x

    Returns
    -------
    np.ndarray
        crop of the input czyx_data given the slicing parameters
    """
    czyx_data_sliced = czyx_data[
        :,
        czyx_slicing_params[0],
        czyx_slicing_params[1],
        czyx_slicing_params[2],
    ]
    return czyx_data_sliced


def append_channels(input_data_path: Path, target_data_path: Path) -> None:
    """
    Append channels to a target zarr store

    Parameters
    ----------
    input_data_path : Path
        input zarr path = /input.zarr
    target_data_path : Path
        target zarr path  = /target.zarr
    """
    appending_dataset = open_ome_zarr(input_data_path, mode="r")
    appending_channel_names = appending_dataset.channel_names
    with open_ome_zarr(target_data_path, mode="r+") as dataset:
        target_data_channel_names = dataset.channel_names
        num_channels = len(target_data_channel_names) - 1
        print(f"channels in target {target_data_channel_names}")
        print(f"adding channels {appending_channel_names}")
        for name, position in tqdm(dataset.positions(), desc="Positions"):
            for i, appending_channel_idx in enumerate(
                tqdm(appending_channel_names, desc="Channel", leave=False)
            ):
                position.append_channel(appending_channel_idx)
                position["0"][:, num_channels + i + 1] = appending_dataset[str(name)][0][:, i]
        dataset.print_tree()
    appending_dataset.close()


def model_to_yaml(model, yaml_path: Path) -> None:
    """
    Save a model's dictionary representation to a YAML file.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    model : object
        The model object to convert to YAML.
    yaml_path : Path
        The path to the output YAML file.

    Raises
    ------
    TypeError
        If the `model` object does not have a `dict()` method.

    Notes
    -----
    This function converts a model object into a dictionary representation
    using the `dict()` method. It removes any fields with None values before
    writing the dictionary to a YAML file.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = MyModel()
    >>> model_to_yaml(model, 'model.yaml')

    """
    yaml_path = Path(yaml_path)

    if not hasattr(model, "dict"):
        raise TypeError("The 'model' object does not have a 'dict()' method.")

    model_dict = model.dict()

    # Remove None-valued fields
    clean_model_dict = {key: value for key, value in model_dict.items() if value is not None}

    with open(yaml_path, "w+") as f:
        yaml.dump(clean_model_dict, f, default_flow_style=False, sort_keys=False)


def yaml_to_model(yaml_path: Path, model):
    """
    Load model settings from a YAML file and create a model instance.

    Borrowing from recOrder==0.4.0

    Parameters
    ----------
    yaml_path : Path
        The path to the YAML file containing the model settings.
    model : class
        The model class used to create an instance with the loaded settings.

    Returns
    -------
    object
        An instance of the model class with the loaded settings.

    Raises
    ------
    TypeError
        If the provided model is not a class or does not have a callable constructor.
    FileNotFoundError
        If the YAML file specified by `yaml_path` does not exist.

    Notes
    -----
    This function loads model settings from a YAML file using `yaml.safe_load()`.
    It then creates an instance of the provided `model` class using the loaded settings.

    Examples
    --------
    >>> from my_model import MyModel
    >>> model = yaml_to_model('model.yaml', MyModel)

    """
    yaml_path = Path(yaml_path)

    if not callable(getattr(model, "__init__", None)):
        raise TypeError("The provided model must be a class with a callable constructor.")

    try:
        with open(yaml_path, "r") as file:
            raw_settings = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The YAML file '{yaml_path}' does not exist.")

    return model(**raw_settings)
