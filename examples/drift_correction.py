# %%
from copylot.hardware.stages.thorlabs.KIM001 import KCube_PiezoInertia
import time
from copylot import logger
# from waveorder.focus import focus_from_transverse_band


def test_labelfree_stage():
    with KCube_PiezoInertia(serial_number='74000565', simulator=False) as stage_LF:
        print(f'LF current position {stage_LF.position}')
        stage_LF.move_relative(10)


def test_light_sheet_stage():
    ### LIGHT SHEET STAGE
    with KCube_PiezoInertia(serial_number='74000291', simulator=False) as stage_LS:

        print(f'LS current position {stage_LS.position}')

        # Change the acceleration and step rate
        stage_LS.step_rate = 500
        stage_LS.step_acceleration = 1000
        print(f'acceleration {stage_LS.step_acceleration} rate {stage_LS.step_rate}')

        # Test relative movement
        step_size = 10
        stage_LS.move_relative(step_size)
        stage_LS.move_relative(-step_size)


if __name__ == '__main__':
    test_light_sheet_stage()
