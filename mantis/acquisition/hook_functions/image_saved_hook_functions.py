from . import config

def check_ls_acq_finished(axes, dataset):
    if axes==config.ls_last_img_idx:
        config.ls_acq_finished = True

def check_lf_acq_finished(axes, dataset):
    if axes==config.lf_last_img_idx:
        config.lf_acq_finished = True