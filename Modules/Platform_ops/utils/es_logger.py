import sys
import traceback
from datetime import datetime

from elasticsearch import Elasticsearch
# from util.logger import getLogger as get_sys_logger

from config.config import config


class EsLogger:
    def __init__(self, index, other_info) -> None:
        self.es = Elasticsearch([config.ES], http_auth=(config.ES_USER, config.ES_PASS))
        self.other_info = other_info
        self.index = index
        # self.logger = get_sys_logger("ESLOGGER_" + index)

    def info(self, msg):
        self.write(msg, "INFO")

    def warn(self, msg):
        self.write(msg, "WARN")

    def error(self, msg):
        self.write(msg, "ERROR")

    def write(self, msg, level="INFO"):
        # self.logger.info(msg)
        msg = str(msg)
        if not msg.strip():
            return
        sys.__stdout__.write(msg + "\n")
        try:
            msg_item = {
                "level": level,
                "msg": msg,
                "timestamp": datetime.utcnow(),
                **self.other_info
            }
            self.es.index(index=self.index, body=msg_item)
        except Exception as e:
            traceback.print_stack()
            sys.__stdout__.write(e + "\n")


def getLogger(index, other_info):
    logger = EsLogger(index, other_info)
    return logger

