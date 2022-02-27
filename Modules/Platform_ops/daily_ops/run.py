# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
import datetime
import traceback
import sys
from os import path
from data_ops.cache.barra_cache.update.update_base_lib import update_base_lib
from DevTools.tools.ding import Ding
from DevTools.tools.elk_logger import get_elk_logger


sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from config.config import config

logger = get_elk_logger("generate_cache_data")
ding = Ding(config.DING_SECRET_KEY, config.DING_ACCESS_TOKEN)
from database import mysql


# 主方法生成缓存文件
def main():
    exec_date = datetime.date.today().strftime('%Y%m%d')
    try:
        #拉取全部数据
        # BaseLibOps().generate_cache_data()
        #更新数据
        update_base_lib().update_data()
        ding.send_ding("INFO |  Barra cache",
                       f"Success | Base_lib data have been generated, date: {exec_date}")
        logger.info(f'Barra cache | Success | Base_lib data have been generated, date: {exec_date}')

    except Exception as e:
        ding.send_ding("ERROR | Barra cache", f'Error | Generation of generate_cache_data fails, '
                                           f'date:{exec_date} | Err-msg: {traceback.format_exc()}')
        logger.error(f'Barra cache | Error | Generation of generate_cache_data fails, '
                     f'date:{exec_date} | Err-msg: {e}')


if __name__ == '__main__':
    main()
