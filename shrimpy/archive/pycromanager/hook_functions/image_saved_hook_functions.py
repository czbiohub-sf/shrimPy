from . import globals


def check_ls_acq_finished(axes, dataset):
    if axes == globals.ls_last_img_idx:
        globals.ls_acq_finished = True


def check_lf_acq_finished(axes, dataset):
    if axes == globals.lf_last_img_idx:
        globals.lf_acq_finished = True
