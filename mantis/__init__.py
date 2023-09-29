import logging

__version__ = "0.1.0"
__mm_version__ = "2023-08-07"


# Define logging console handler
def get_console_handler():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = get_console_formatter()
    console_handler.setFormatter(console_format)
    return console_handler


def get_console_formatter():
    console_format = logging.Formatter('%(levelname)s - %(module)s.%(funcName)s - %(message)s')
    return console_format


# Setup logger
logger = logging.getLogger('mantis')
logger.setLevel(logging.DEBUG)

logger.addHandler(get_console_handler())
logger.propagate = False
