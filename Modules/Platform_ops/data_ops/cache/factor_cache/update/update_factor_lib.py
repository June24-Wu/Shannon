import datetime
import os

import pandas as pd
from DBUtils.PooledDB import PooledDB
import pymysql
from multiprocessing import Pool

from config.config import FACTOR_LIB_TI0_MYSQL

factor_lib = r'/home/ShareFolder/lib_data/factor_lib'

pool = PooledDB(pymysql, 24, **FACTOR_LIB_TI0_MYSQL)


def get_tables(status=None):
    conn = pool.connection()
    curs = conn.cursor()
    if status is not None:
        sql = f"select table_name from table_info where status='{status}';"
    else:
        sql = "select table_name from table_info;"

    tables = []
    try:
        curs.execute(sql)
        result = curs.fetchall()
        tables = [table[0] for table in result if table[0] != 'table_info']
    except Exception as e:
        print(e)

    conn.close()
    return tables


def save_to_file(table):
    conn = pool.connection()

    path = os.path.join(factor_lib, table)

    if not os.path.exists(path):
        os.makedirs(path)

    old_files = os.listdir(path)

    if old_files:
        old_filename = old_files[0]
        old_filepath = os.path.join(path, old_files[0])
        last_update_time = old_filename.split('.')[0]
        if old_filepath.endswith('.par'):
            old_df = pd.read_parquet(old_filepath)
            if not old_df.empty:
                sql = f"""
                        SELECT symbol, trading_time, alpha_value, create_time FROM {table} 
                        where trading_time > '{last_update_time}' ORDER BY trading_time, alpha_value DESC;"""
                df = pd.read_sql(sql, conn)

                last_time = df['trading_time'].iloc[-1]
                int_last_time = int(datetime.datetime.strftime(last_time, '%Y%m%d%H%M%S'))

                df = df.set_index(['trading_time', 'symbol'])
                df = pd.concat([old_df, df])

                print(f'{datetime.datetime.now()} -- {path}/{int_last_time}.par')
                os.remove(old_filepath)
                df.to_parquet(f'{path}/{int_last_time}.par')
                conn.close()
    else:
        sql = f"""
                SELECT symbol, trading_time, alpha_value, create_time FROM {table} 
                ORDER BY trading_time, alpha_value DESC;"""

        df = pd.read_sql(sql, conn)
        last_time = df['trading_time'].iloc[-1]
        int_last_time = int(datetime.datetime.strftime(last_time, '%Y%m%d%H%M%S'))

        df = df.set_index(['trading_time', 'symbol'])
        print(f'{datetime.datetime.now()} -- {path}/{int_last_time}.par')
        df.to_parquet(f'{path}/{int_last_time}.par')
        conn.close()


def main():
    tables = get_tables()
    process_pool = Pool(24)
    process_pool.map(save_to_file, tables)
    process_pool.join()
    process_pool.close()


if __name__ == '__main__':
    main()
