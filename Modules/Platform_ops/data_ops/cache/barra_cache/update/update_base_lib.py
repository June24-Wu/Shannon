import datetime
import os
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))))
import pandas as pd
import pymysql
from DBUtils.PooledDB import PooledDB
from multiprocessing import Pool
from config.config import BASE_LIB_MYSQL


base_path = r'/home/ShareFolder/lib_data/base_lib'


tables = [
    "factor_returns_hs300",
    "factor_returns_investable",
    "factor_returns_zz500",
    "factor_exposure_hs300_hs300",
    "factor_exposure_investable_zz800",
    "factor_exposure_zz500_zz500",
]

return_tables = [
    "factor_returns_hs300",
    "factor_returns_investable",
    "factor_returns_zz500"
]

#更新数据
class update_base_lib(object):
    def __init__(self):
        self.pool = PooledDB(pymysql, 6, **BASE_LIB_MYSQL)

    def update_data(self):
        conn = self.pool.connection()
        file = os.listdir(os.path.join(base_path, 'factor', 'factor_exposure_hs300_hs300'))[0]
        timeint = int(file.split('.')[0])
        for table in tables:
            sql = f"""select * from {table} where dt_int > {timeint};"""
            #将dt_index变为索引
            df_db = pd.read_sql(sql, conn).set_index('dt_int')
            print(df_db)
            #测试
            # old_df = pd.read_parquet(os.path.join(base_path,'return','factor_returns_hs300','20211110.par'))
            # old_df = pd.read_parquet(os.path.join(base_path,'return','factor_returns_investable','20211110.par'))
            # old_df = pd.read_parquet(os.path.join(base_path,'return','factor_returns_zz500','20211110.par'))
            # old_df = pd.read_parquet(os.path.join(base_path,'factor','factor_exposure_hs300_hs300','20211110.par'))
            # old_df = pd.read_parquet(os.path.join(base_path,'factor','factor_exposure_investable_zz800','20211110.par'))
            # old_df = pd.read_parquet(os.path.join(base_path,'factor','factor_exposure_zz500_zz500','20211110.par'))
            # old_df = old_df.drop_duplicates()
            # print(old_df.count())
            # if not df_db.values:
            #     return
            # df_db = pd.read_sql(sql, conn)
            if table in return_tables:
                path = os.path.join(base_path, 'return', table)
            else:
                path = os.path.join(base_path, 'factor', table)
            last_time = df_db.index[-1]
            str_last_time = str(last_time)
            old_df = pd.read_parquet(path)
            # if not os.path.exists(path):
            #     os.makedirs(path)
            if os.listdir(path):
                os.system(f'cd {path} && rm *.par')
            # dt_int index
            new_df = pd.concat([df_db, old_df])
            new_df.drop(columns=['dt'], inplace=True)
            new_df = new_df.drop_duplicates()
            if 'symbol' in new_df.columns:
                new_df = new_df.sort_values(by=['dt_int', 'symbol'])
            print(f'{path}/{str_last_time}.par')
            new_df.to_parquet(f'{path}/{str_last_time}.par')
        conn.close()




if __name__ == '__main__':
    update_base_lib().update_data()