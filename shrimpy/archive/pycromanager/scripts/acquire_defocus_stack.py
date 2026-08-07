import numpy as np
from pycromanager import Core, Studio
from mantis.acquisition.microscope_operations import (
    setup_kim101_stage, 
    acquire_ls_defocus_stack_and_display,
    set_relative_kim101_position,
)
from waveorder.focus import focus_from_transverse_band

mmc = Core()
mmStudio = Studio()
z_start = -105
z_end = 105
z_step = 15
galvo = 'AP Galvo'
galvo_range = [-0.5, 0, 0.5]

z_stage = setup_kim101_stage('74000291')
z_range = np.arange(z_start, z_end + z_step, z_step)

# run 5 times over
for i in range(5):
    data = acquire_ls_defocus_stack_and_display(
        mmc, 
        mmStudio, 
        z_stage, 
        z_range,
        galvo,
        galvo_range,
        close_display=False,
    )

    focus_indices = []
    for stack in data:
        idx = focus_from_transverse_band(
            stack, NA_det=1.35, lambda_ill=0.55, pixel_size=6.5/40/1.4
        )
        focus_indices.append(idx)

    valid_focus_indices = [idx for idx in focus_indices if idx is not None]
    print(f'Valid focus indices: {valid_focus_indices}')

    focus_idx = int(np.median(valid_focus_indices))
    o3_displacement = int(z_range[focus_idx])
    print(f'O3 displacement: {o3_displacement} steps')

    set_relative_kim101_position(z_stage, o3_displacement)

