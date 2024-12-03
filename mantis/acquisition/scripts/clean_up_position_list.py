import numpy as np
import pandas as pd

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

well_diameter = 4.0  # in mm
min_fov_distance_from_well_edge = 0.5

xyz_positions, position_labels = get_position_list(
    mmStudio,
    z_stage_name=None,
)
xyz_positions = np.asarray(xyz_positions, dtype=np.float64)
position_labels = np.asarray(position_labels)
well_names = [position_label.split('-')[0] for position_label in position_labels]
fov_names = [position_label.split('-')[1] for position_label in position_labels]
wells_labels_positions = pd.DataFrame({
    'well': well_names,
    'fov': fov_names,
    'x-position': [p[0] for p in xyz_positions],
    'y-position': [p[1] for p in xyz_positions],
})

pos_list_manager = mmStudio.get_position_list_manager()
mm_pos_list = pos_list_manager.get_position_list()

# assume FOVs are evenly distributed around the well center
microns_per_mm = 1000
print(f'Initial total number of positions: {mm_pos_list.get_number_of_positions()} \n')
for well in set(well_names):
    df_well = wells_labels_positions[wells_labels_positions['well']==well]
    well_center = df_well[['x-position', 'y-position']].mean(axis=0)
    fov_distance_from_center = np.sqrt(((df_well[['x-position', 'y-position']] - well_center) ** 2).sum(axis=1)) / microns_per_mm
    fovs_to_remove = df_well['fov'][fov_distance_from_center > (well_diameter / 2 - min_fov_distance_from_well_edge)]

    print(f'Initial number of positions in well {well}: {len(df_well)}')
    print(f'FOVs which will be removed: {fovs_to_remove.values}')

    for fov in fovs_to_remove:
        position_label = '-'.join((well, fov))
        mm_pos_list.remove_position(mm_pos_list.get_position_index(position_label))

    print(f'Number of positions remaining in well {well}: {len(df_well) - len(fovs_to_remove)}\n')

print(f'Remaining total number of positions: {mm_pos_list.get_number_of_positions()}')
pos_list_manager.set_position_list(mm_pos_list)
