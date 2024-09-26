import numpy as np

from pycromanager import Studio


# Copied from mantis/acquisition/microscope_operations.py
def get_position_list(mmStudio, z_stage_name):
    mm_pos_list = mmStudio.get_position_list_manager().get_position_list()
    number_of_positions = mm_pos_list.get_number_of_positions()

    xyz_positions = []
    position_labels = []
    for i in range(number_of_positions):
        _pos = mm_pos_list.get_position(i)
        xyz_positions.append(
            [
                _pos.get_x(),
                _pos.get_y(),
                _pos.get(z_stage_name).get1_d_position() if z_stage_name else None,
            ]
        )
        position_labels.append(_pos.get_label())

    return xyz_positions, position_labels


mmStudio = Studio()

well_diameter = 35.0  # in mm
min_fov_distance_from_well_edge = 0.5

xyz_positions, position_labels = get_position_list(
    mmStudio,
    z_stage_name=None,
)
xyz_positions = np.asarray(xyz_positions, dtype=np.float64)
position_labels = np.asarray(position_labels)

# assume FOVs are evenly distributed around the well center
microns_per_mm = 1000
well_center = xyz_positions.mean(axis=0)[:2]
fov_distance_from_center = (
    np.sqrt(((xyz_positions[:, :2] - well_center) ** 2).sum(axis=1)) / microns_per_mm
)

positions_to_remove = position_labels[
    fov_distance_from_center > (well_diameter / 2 - min_fov_distance_from_well_edge)
]

pos_list_manager = mmStudio.get_position_list_manager()
mm_pos_list = pos_list_manager.get_position_list()

print(f'Initial number of positions: {mm_pos_list.get_number_of_positions()}')
print(f'Positions which will be removed: {positions_to_remove}')

for position in positions_to_remove:
    mm_pos_list.remove_position(mm_pos_list.get_position_index(position))

print(f'Number of positions remaining: {mm_pos_list.get_number_of_positions()}')
pos_list_manager.set_position_list(mm_pos_list)
