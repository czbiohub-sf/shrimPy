from pycromanager import Dataset
import numpy as np
import napari
import time
from napari.qt import thread_worker

LF_LAYER_NAME = 'label-free data'
LS_LAYER_NAME = 'light-sheet data'

lf_dataset_path = r'Z:\rawdata\mantis\2023_04_20 HEK RAC1 PCNA\1timepoint_test_1\1timepoint_test_labelfree_1'
ls_dataset_path = r'Z:\rawdata\mantis\2023_04_20 HEK RAC1 PCNA\1timepoint_test_1\1timepoint_test_lightsheet_1'

viewer = napari.Viewer()
lf_dataset = Dataset(lf_dataset_path)
ls_dataset = Dataset(ls_dataset_path)

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
def napari_signaller(lf_dataset, ls_dataset):
    """
    Monitor for signals that Acqusition has a new image ready, and when that happens
    update napari appropriately
    """
    while True:
        if time.time() - t_start < 60:
            print('waiting 1 sec')
            time.sleep(1)
        elif time.time() - t_start < 5 * 60:
            print('waiting 15 sec')
            time.sleep(15)
        elif time.time() - t_start < 60 * 60:
            print('waiting 2 min')
            time.sleep(2 * 60)
        else:
            print('waiting 10 min')
            time.sleep(10 * 60)

        lf_data, ls_data = None, None
        if lf_dataset is not None and ls_dataset is not None:
            lf_data = lf_dataset.as_array()
            ls_data = ls_dataset.as_array()

        yield (lf_data, ls_data)

napari_signaller(lf_dataset, ls_dataset)
napari.run()