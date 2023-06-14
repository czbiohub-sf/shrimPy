import click
from iohub import open_ome_zarr
from pathlib import Path
import datetime
import os
import glob
from mantis.cli.deskew import deskew_cli, get_deskew_params, create_empty_zarr, get_output_paths
import yaml
from dataclasses import asdict
import multiprocessing as mp
from slurmkit import SlurmParams, slurm_function, submit_function
from iohub.ngff_meta import TransformationMeta
from natsort import natsorted

#TODO: decide what logging we use (i.e click.echo, print, logging)


@slurm_function
def slurm_deskew_cli(
    input_data_path: Path,
    deskew_params_path: Path,
    output_path: Path,
    view: bool,
    keep_overhang: bool,
    num_cores: int,
) -> None:
    """ Wraps the deskew function by  """

    deskew_cli(
        input_data_path=input_data_path,
        deskew_params_path=deskew_params_path,
        output_path=output_path,
        view=view,
        keep_overhang=keep_overhang,
        num_cores=num_cores,
    )

def slurm_deskew(
    input_data_path:Path,
    deskew_param_path:Path = "./deskew_settings.yml",
    output_path:Path = './deskew.zarr',
    view:bool = False,
    keep_overhang:bool = False,
    num_cores:int = mp.cpu_count()
):
    #---------------INITIALIZATION AND DATA PARSING ----------------------
    # Deskewing I/O
    input_data_path = glob.glob(input_data_path)
    input_data_path = natsorted(input_data_path)
    output_dir = os.path.dirname(os.path.join(os.getcwd(), output_path))
    list_output_pos = get_output_paths(input_data_path, output_path)
    click.echo(f"in: {input_data_path}, out: {list_output_pos}")
    #Slurm I/O
    slurm_out_path = str(os.path.join(output_dir, "slurm_output/deskew-%j.out"))
    
    # Initialize the array
    create_empty_zarr(input_data_path, deskew_param_path, output_path, keep_overhang)

    #-------------------- SLURM PARAMETERS ----------------------------------

    # Slurm Parameters for each node
    # NOTE: These parameters have to be tunned depending on the size of the job
    params = SlurmParams(
        partition="cpu",
        cpus_per_task=16,
        mem="64G",
        time=datetime.timedelta(minutes=40),
        output=slurm_out_path,
    )
    #-----------------------------------------------------------------
    # Wrap our deskew_cli() function with slurmkit
    deskew_func = slurm_deskew_cli(
        deskew_params_path=deskew_param_path,
        view=view,
        keep_overhang=keep_overhang,
        num_cores=num_cores
    )
    # Generates the array of jobs by passing the in_path and out_path to slurm wrapped function
    deskew_jobs = [
        submit_function(
            deskew_func,
            slurm_params=params,
            input_data_path=in_path,
            output_path=out_path,
        )
        for in_path, out_path in zip(input_data_path, list_output_pos)
    ]

if __name__ == "__main__":
    input_data_path = '/hpc/projects/comp.micro/mantis/2023_05_10_PCNA_RAC1/0-crop-convert-zarr/sample_short.zarr/*/*/*'
    output_data_path = './deskewed.zarr'
    deskew_param_path = './deskew_settings.yml'

    #Execute SLURM with deskewing
    slurm_deskew(input_data_path, deskew_param_path, output_data_path)
