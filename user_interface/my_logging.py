import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def get_logger(module_name):
    return logger.getChild(module_name)


ERROR_PATH = Path('logs/errors')
if not ERROR_PATH.exists():
    ERROR_PATH.mkdir(parents=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

handler = TimedRotatingFileHandler(
    f'{ERROR_PATH}/error_log',
    when='midnight',
    interval=1,
    backupCount=14,
    encoding='utf-8'
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger.addHandler(handler)