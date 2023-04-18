import os
import csv
from mantis.acquisition.BaseSettings import PositionSettings

position_settings = PositionSettings(**{})

pos_list = []
for i in range(position_settings.num_positions):
    pos_list.append(
        {
            'position_ind': i,
            'well_id': position_settings.position_labels[i].split('-')[0],
            'site_num': position_settings.position_labels[i].split('_')[-1],
            'label': position_settings.position_labels[i],
            'x_position': position_settings.xyz_positions[i][0],
            'y_position': position_settings.xyz_positions[i][1],
            'z_position': position_settings.xyz_positions[i][2],
        }
    )

path = r'Z:\rawdata\mantis\2023_04_05_mantis_HEK\48wells_1timepoint_4'
with open(os.path.join(path, 'positions.csv'), 'w', newline='') as csvfile:
    fieldnames = list(pos_list[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(pos_list)