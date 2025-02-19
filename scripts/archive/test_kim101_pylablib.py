from pylablib.devices import Thorlabs
devices = Thorlabs.list_kinesis_devices()

stage = Thorlabs.KinesisPiezoMotor('74000291')

p = stage.get_position()
for i in range(50):
    # stage.move_to(p+25); stage.wait_move()
    # stage.move_to(p-25); stage.wait_move()

    # relative moves work better
    stage.move_by(25); stage.wait_move()
    stage.move_by(-25); stage.wait_move()

print('done')
