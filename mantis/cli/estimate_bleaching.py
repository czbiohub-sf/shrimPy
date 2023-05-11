import click
import matplotlib.pyplot as plt

from iohub import read_micromanager


@click.command()
@click.argument(
    "data_path",
    type=click.Path(exists=True),
)
@click.option(
    "--output-file",
    "-o",
    default="./bleaching.pdf",
    required=False,
    help="Path to saved parameters",
)
def estimate_bleaching(data_path, output_file):
    """Estimate bleaching from raw data"""
    # Read data
    reader = read_micromanager(data_path)
    import pdb

    for p in range(reader.get_num_positions()):
        for t in range(reader.shape[0]):
            for c in range(reader.shape[1]):
                time = reader.get_image_metadata(p, t, c, 0)["TimeStampMsec"]
    pdb.set_trace()
    data = reader.get_array(0)[0, 0, ...]  # zyx

    # Collect statistics for each time point

    # Generate and save plot
