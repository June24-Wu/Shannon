import datetime
import importlib
import os
import sys
import time

import pandas as pd
from Platform.backtest.bt import get_trading_days
import pickle

import redis


class CheckoutDf(object):
    def __init__(self):

        self.host = '0.0.0.0'
        self.port = 6379

    def con_redis(self):
        pool = redis.ConnectionPool(host=self.host, port=self.port)
        r = redis.Redis(connection_pool=pool)
        return r

    def run(self,file_path):
        r = self.con_redis()
        today = datetime.date.today()
        last_date = get_trading_days(today, today)[0] - datetime.timedelta(days=1)
        int_last_day = int(last_date.strftime("%Y%m%d"))
        filename = os.path.basename(file_path)
        dir_path = os.path.dirname(file_path)
        sys.path.append(dir_path)
        class_name = filename.strip('.so')
        module_object = importlib.import_module(class_name)
        module_cls = getattr(module_object, class_name)
        try:
            df_dic = module_cls().create(int_last_day, int_last_day)
            # for k, v in df_dic.items():
            #     v = v.rename(columns={"symbol": "ticker"})
            #     r.set(k, pickle.dumps(v))

        except Exception as e:
            print(e)
            return
        t1 = time.time()
        df_dic = module_cls().create(20160101, int_last_day)
        t2 = time.time()
        for k, v in df_dic.items():
            v = v.rename(columns={"symbol":"ticker"})
            r.set(k,pickle.dumps(v))
        # print(t2-t1)

path = '/home/ShareFolder/factors_ops/Factor_Northflow.so'
# CheckoutDf().run(path)