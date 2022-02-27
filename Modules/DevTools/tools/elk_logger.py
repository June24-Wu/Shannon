import getpass
from logging import Logger
import logging.config
import socket
import os
from logging.handlers import RotatingFileHandler
import logging
import time


def get_elk_logger(name, console=True) -> Logger:
    user = getpass.getuser()
    nPID = os.getpid()

    yearmonth = time.strftime('%Y%m', time.localtime(time.time()))
    LOG_PATH = "/home/ShareFolder/elk_logs"
    LOG_PATH = os.path.join(LOG_PATH, name, yearmonth)
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    # Define your logging settings.
    log_file = os.path.join(LOG_PATH, f'info-{name}-{socket.gethostname()}-{nPID}.log')
    root_logger = logging.getLogger(user)
    if len(root_logger.handlers) > 0:
        return root_logger
    root_logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        f'%(asctime)s | %(levelname)s | {socket.gethostname()} | {user} | {nPID} | %(process)d | '
        f'%(thread)d | %(filename)s | %(module)s | %(funcName)s | %(lineno)d | %(message)s')

    # formatter=logging.Formatter("%(message)s")
    rotating_file_log = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=1)
    rotating_file_log.setLevel(logging.INFO)
    rotating_file_log.setFormatter(formatter)
    root_logger.addHandler(rotating_file_log)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger
