import logging
from mantis.acquisition.logger import configure_logger

configure_logger('logfile.txt')
print(f'{__name__}')
logger = logging.getLogger(__name__)

logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning')
logger.error('This is an error')