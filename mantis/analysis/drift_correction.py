# %%
from copylot.hardware.stages.thorlabs.KIM001 import KCube_PiezoInertia
import time
from copylot import logger

### LABEL_FREE STAGE
# with KCube_PiezoInertia(serial_number='74000565', simulator=False) as stage_LF:
    # print(f'LF current position {stage_LF.position}')
    # stage_LF.move_relative(10)

### LIGHT SHEET STAGE
with KCube_PiezoInertia(serial_number='74000291', simulator=False) as stage_LS:
    # Test the relative movement
    print(f'LS current position {stage_LS.position}')
    stage_LS.move_relative(100)
    stage_LS.move_relative(-100)

    # Change the acceleration and step rate
    stage_LS.step_rate = 50
    stage_LS.step_acceleration = 200
    print(f'acceleration {stage_LS.step_acceleration} rate {stage_LS.step_rate}')
    # Test the movement with the different acceleration
    stage_LS.move_relative(100)
    stage_LS.move_relative(-100)

