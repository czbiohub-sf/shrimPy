# %%
import json

from pathlib import Path
from pycromanager import Core, Studio, JavaObject

position_list_dir = r'G:\OPS\OPS0072_1'
position_list_path = Path(position_list_dir) / 'pheno_position_list.json'

mmc = Core()
mmStudio = Studio()


def populate_position_list(
    mmc,
    mmStudio,
    xyz_positions: list,
    position_labels: list,
    xy_stage_name: str=None,
    z_stage_name: str=None
):
    if xy_stage_name is None:
        xy_stage_name = mmc.get_xy_stage_device()
    if z_stage_name is None:
        z_stage_name = mmc.get_focus_device()

    mm_pos_list_manager = mmStudio.get_position_list_manager()
    mm_pos_list = mm_pos_list_manager.get_position_list()

    for _pos, mm_pos_label in zip(xyz_positions, position_labels):
        mm_position = JavaObject(
            "org.micromanager.MultiStagePosition",
            [
                xy_stage_name,
                _pos[0],
                _pos[1],
                z_stage_name,
                _pos[2]
            ]
        )
        mm_position.set_label(mm_pos_label)
        mm_pos_list.add_position(mm_position)

    # Set the updates position list as the "current" position list
    mm_pos_list_manager.set_position_list(mm_pos_list)
    print('Updated MicroManager position list with new coordinates')

# %%
with open(position_list_path) as file:
    position_list = json.load(file)

populate_position_list(
    mmc,
    mmStudio,
    list(position_list.values()),
    list(position_list.keys()),
    xy_stage_name='XYStage',
    z_stage_name='FocusDrive',
)

# %%
