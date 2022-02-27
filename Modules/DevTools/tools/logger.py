import getpass
import logging
import logging.config
import os
import socket
from datetime import datetime
from logging import Logger
from logging.handlers import RotatingFileHandler

LOG_PATH = "./log"


def get_logger(name, output_path=None, console_flag=True) -> Logger:
    user = getpass.getuser()
    nPID = os.getpid()

    if output_path is None:
        output_path = LOG_PATH

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define your logging settings.
    log_file = os.path.join(LOG_PATH, f'info-{name}-{datetime.now().strftime("%Y-%m-%d")}.log')
    log_file_err = os.path.join(LOG_PATH, f'info-{name}-{datetime.now().strftime("%Y-%m-%d")}-error.log')
    root_logger = logging.getLogger(name)

    if len(root_logger.handlers) > 0:
        return root_logger

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        f'%(asctime)s | %(levelname)s | {socket.gethostname()} | {user} | {nPID} | %(process)d | '
        f'%(thread)d | %(filename)s | %(module)s | %(funcName)s | %(lineno)d | %(message)s')

    rotating_file_log = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=1)
    rotating_file_log.setLevel(logging.INFO)
    rotating_file_log.setFormatter(formatter)
    root_logger.addHandler(rotating_file_log)

    rotating_file_log_err = RotatingFileHandler(log_file_err, maxBytes=10485760, backupCount=1)
    rotating_file_log_err.setLevel(logging.ERROR)
    rotating_file_log_err.setFormatter(formatter)
    root_logger.addHandler(rotating_file_log_err)

    if console_flag:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger
