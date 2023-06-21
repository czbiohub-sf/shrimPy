# %%
from copylot.hardware.stages.thorlabs.KIM001 import KCube_PiezoInertia
import time
from copylot import logger

### LABEL_FREE STAGE
# stage_LF = KCube_PiezoInertia(serial_number='74000565', simulator=False)

# Instantiate the piezo stages
with KCube_PiezoInertia(serial_number='74000291', simulator=False) as stage_LS:
    # Test Moving
    # print(f'LF current position {stage_LF.position}')
    # stage_LF.move_relative(10)
    print(f'LS current position {stage_LS.position}')
    stage_LS.move_relative(100)
    stage_LS.move_relative(-100)
    stage_LS.step_rate = 50
    stage_LS.step_acceleration = 200
    print(f'acceleration {stage_LS.step_acceleration} rate {stage_LS.step_rate}')
    stage_LS.move_relative(100)
    stage_LS.move_relative(-100)

# stage_LS = KCube_PiezoInertia(serial_number='74000291', simulator=False)
# print(f'LS current position {stage_LS.position}')
# stage_LS.move_relative(100)
# stage_LS.move_relative(-100)
# stage_LS.disconnect()