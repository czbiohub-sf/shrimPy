from pycromanager import Core, start_headless

PORT2 = 4927
mm_app_path = r'C:\\Program Files\\Micro-Manager-nightly'
config_file = r'C:\\CompMicro_MMConfigs\\shrimpy\\shrimpy-LS.cfg'

print('Starting headless mode')
start_headless(mm_app_path, config_file, port=PORT2)
print('Started headless mode')

print('Connecting to core')
mmc = Core(port=PORT2)
print('Connected to core')

print('Setting blanking mode')
mmc.set_property('TS2_TTL1-8', 'Blanking', 'On')

print('done')
