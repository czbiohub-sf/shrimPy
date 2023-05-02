import click
import os
from pycromanager import Dataset
import napari
import time
from napari.qt import thread_worker

_verbose = False
LF_LAYER_NAME = 'label-free data'
LS_LAYER_NAME = 'light-sheet data'

viewer = napari.Viewer()
t_start = time.time()

def update_layers(data):
    """
    update the napari layer with the new image
    """
    lf_data = data[0]
    ls_data = data[1]

    if lf_data is not None and ls_data is not None:
        # update data
        try:
            viewer.layers[LF_LAYER_NAME].data = lf_data
            viewer.layers[LS_LAYER_NAME].data = ls_data
        # layers do not exist, create display
        except KeyError:
            viewer.add_image(lf_data, name=LF_LAYER_NAME)
            viewer.add_image(ls_data, name=LS_LAYER_NAME)

@thread_worker(connect={'yielded': update_layers})
def napari_signaller(lf_dataset_path, ls_dataset_path):
    """
    Monitor for signals that Acquisition has a new image ready, and when that happens
    update napari appropriately
    """
    while True:
        lf_dataset = Dataset(lf_dataset_path)
        ls_dataset = Dataset(ls_dataset_path)

        lf_data = lf_dataset.as_array()
        ls_data = ls_dataset.as_array()
        if _verbose: print(f'LF data shape: {lf_data.shape}, LF data shape: {ls_data.shape}')

        yield (lf_data, ls_data)

        if time.time() - t_start < 60:
            if _verbose: print('waiting 1 sec')
            time.sleep(5)
        elif time.time() - t_start < 5 * 60:
            if _verbose: print('waiting 15 sec')
            time.sleep(15)
        elif time.time() - t_start < 60 * 60:
            if _verbose: print('waiting 2 min')
            time.sleep(2 * 60)
        else:
            if _verbose: print('waiting 10 min')
            time.sleep(10 * 60)

        if _verbose: print('Closing datasets')
        lf_dataset.close()
        ls_dataset.close()


@click.command()
@click.argument(
    "dataset_path",
    type=click.Path(exists=True, file_okay=False),
)
def run_viewer(dataset_path):
    dirname, basename = os.path.split(dataset_path)

    lf_dataset_name = '_'.join(basename.split('_')[:-1]) + '_labelfree_1'
    lf_dataset_path = os.path.join(dirname, basename, lf_dataset_name)

    ls_dataset_name = '_'.join(basename.split('_')[:-1]) + '_lightsheet_1'
    ls_dataset_path = os.path.join(dirname, basename, ls_dataset_name)

    napari_signaller(lf_dataset_path, ls_dataset_path)
    napari.run()

if __name__ == '__main__':
    run_viewer()
