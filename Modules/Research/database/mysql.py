import datetime
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pymysql
from DBUtils.PooledDB import PooledDB
from DataAPI import convert_to_datetime
from ExObject.DateTime import DateTime

from ..config.config import LIB_PATH, BASE_LIB_MYSQL, TEST_FACTORS_DAILY_TI0, FACTOR_LIB_MYSQL_TIO

factor_lib_path = os.path.join(LIB_PATH, 'factor_lib')
base_lib_path = os.path.join(LIB_PATH, 'base_lib')


class MysqlAPI(object):
    def __init__(self, mysql_info=FACTOR_LIB_MYSQL_TIO, threads=1, logger=None):
        """
        :param mysql_info: your mysql account, password, host, port,database name...
        :param logger: the logger file path
        """
        self.pool = PooledDB(pymysql, threads, **mysql_info)
        self.logger = logger

    # 5. get status before query db
    def query(self, table_name, start_time, end_time):
        try:
            start_time = convert_to_datetime(start_time).strftime('%Y-%m-%d 00:00:00')
            end_time = convert_to_datetime(end_time).strftime('%Y-%m-%d 00:00:00')
        except Exception as e:
            print(f"Error, start time or end time is invalid, detail is {e}.")
            return pd.DataFrame()

        conn = self.pool.connection()

        sql = f"""select * from {table_name} where trading_time >= '{start_time}' and trading_time <='{end_time}';"""

        try:
            df = pd.read_sql(sql, conn)
        except Exception as e:
            if self.logger is not None:
                self.logger.error(f"""remove error, error is {e}.""")
            conn.rollback()
            raise e

        conn.close()

        return df

    def get_tables(self, status=None):
        conn = self.pool.connection()
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

    def get_factor_info(self):
        conn = self.pool.connection()
        sql = f"""select * from table_info order by create_time asc;"""
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_factor_rejected_info(self):
        conn = self.pool.connection()
        sql = f"""select * from table_info_rejected;"""
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def query_factor(self, name, table_name, start_time, end_time):
        try:
            s_time = convert_to_datetime(str(start_time) + '093000')
            e_time = convert_to_datetime(str(end_time) + '093000')

            str_s_time = s_time.strftime("%Y-%m-%d %H:%M:%S")
            str_e_time = e_time.strftime("%Y-%m-%d %H:%M:%S")

            file_path = os.path.join(factor_lib_path, name, table_name)
            file_name = os.listdir(file_path)[0]
            table_time = int(file_name.split('.')[0])
            file_date = table_time
            table_time = str(table_time) + '093000'
            str_table_time = str(datetime.datetime.strptime(str(table_time), "%Y%m%d%H%M%S"))
            table_path = os.path.join(file_path, file_name)
            if not os.path.exists(table_path):
                print('Error, table is not exists.')
                return pd.DataFrame()

            df_par = pd.read_parquet(table_path)

            # s_time_date = str_s_time.split(' ')[0]
            if df_par.index.get_level_values(0)[0] > pd.Timestamp(s_time):
                print('Error, the start time is too early.')
                return pd.DataFrame()
            if int(start_time) > file_date:
                sql = f"""select symbol, trading_time, alpha_value, create_time from {table_name} where trading_time >= '{str_s_time}' and trading_time <= '{str_e_time}';"""
                conn = self.pool.connection()

                df_db = pd.read_sql(sql, conn).set_index(['trading_time', 'symbol'])
                print(df_db)
                return df_db

            if int(end_time) > file_date:
                df_par = df_par.loc[str_s_time:str_table_time]
                sql = f"""select symbol, trading_time, alpha_value, create_time from {table_name} where trading_time > '{str_table_time}';"""
                conn = self.pool.connection()
                df_db = pd.read_sql(sql, conn).set_index(['trading_time', 'symbol'])
                conn.close()
                df_par = pd.concat([df_par, df_db])
                print(df_par)
            else:
                df_par = df_par.loc[str_s_time:str_e_time]
            print(df_par)
            return df_par

        except Exception as e:
            print(f'Error, details is {e}.')

    def query_factor_ic(self, name, start_time, end_time, benchmark='Investable', price='close', period=('1d',)):
        conn = self.pool.connection()
        query_list = [f"IC_{item}" for item in period]
        query_str = ','.join(query_list)
        sql = f"""
        select trading_time, {query_str} from alpha_ic_daily_{benchmark.lower()}_{price} where alpha_name='{name}' and 
        trading_time >= '{str(start_time)}' and trading_time <= '{str(end_time)}';
        """
        try:
            df = pd.read_sql(sql, conn)
            conn.close()
            return df
        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()

    def query_factor_return(self, name, start_time, end_time, price='close', group=10):
        conn = self.pool.connection()
        query_list = [f"alpha_group{item}" for item in range(group)]
        query_str = ','.join(query_list)
        sql = f"""
        select trading_time, {query_str} from factors_return_daily_group{group}_{price} where alpha_name='{name}' and 
        trading_time >= '{str(start_time)}' and trading_time <= '{str(end_time)}';
        """
        try:
            df = pd.read_sql(sql, conn)
            conn.close()
            return df
        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()

    def query_factor_rec_ic(self, name, benchmark='Investable', period=('1y',)):
        conn = self.pool.connection()
        query_list = [f"rec_{item}_IC" for item in period]
        query_str = ','.join(query_list)
        sql = f"select {query_str} from alpha_icir_info_{benchmark.lower()} where alpha_name='{name}';"
        try:
            df = pd.read_sql(sql, conn)
            conn.close()
            if len(query_list) == 1:
                return df[query_list[0]].values[0]
            else:
                return df[query_list].values[0]
        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()

    def query_factor_rec_return(self, name, benchmark='Investable', period=('1y',)):
        conn = self.pool.connection()
        query_list = [f"rec_{item}_ls_ret_afc_1" for item in period]
        query_str = ','.join(query_list)
        sql = f"select {query_str} from alpha_return_after_cost_info_{benchmark.lower()} where alpha_name='{name}';"
        try:
            df = pd.read_sql(sql, conn)
            conn.close()
            if len(query_list) == 1:
                return df[query_list[0]].values[0]
            else:
                return df[query_list].values[0]

        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()


class BaseLib(object):
    def __init__(self, mysql_info=BASE_LIB_MYSQL, logger=None):
        self.logger = logger
        self.pool = PooledDB(pymysql, 24, **mysql_info)

    def get_tables(self):
        sql = "show tables;"
        conn = self.pool.connection()
        curs = conn.cursor()

        try:
            curs.execute(sql)
            res = curs.fetchall()
            tables = [table[0] for table in res]

        except Exception as e:
            raise e

        else:
            return tables

        finally:
            conn.close()

    def query_barra_factors(self, table_name, start_time, end_time):
        try:
            s_time = int(convert_to_datetime(start_time).strftime('%Y%m%d'))
            e_time = int(convert_to_datetime(end_time).strftime('%Y%m%d'))
        except Exception as e:
            print(f"Error, start time or end time is invalid, detail is {e}.")
            return pd.DataFrame()

        file_path = os.path.join(base_lib_path, 'factor', table_name)
        file_name = os.listdir(file_path)[0]
        table_time = int(file_name.split('.')[0])
        table_path = os.path.join(file_path, file_name)

        if not os.path.exists(table_path):
            print('Error, table is not exists.')
            return pd.DataFrame()

        df_par = pd.read_parquet(table_path)
        if df_par.index[0] > s_time:
            print('Error, the start time is too early.')
            return pd.DataFrame()

        if e_time <= table_time:
            df_par = df_par.loc[(s_time <= df_par.index) & (e_time >= df_par.index)]
            return df_par

        # df_par = df_par.loc[s_time <= df_par.index]
        df_par = df_par.loc[s_time:df_par.index[-1]]
        if table_time < s_time:
            sql = f"""select * from {table_name} where dt_int >= '{s_time}' and dt_int <= '{e_time}';"""
        else:
            sql = f"""select * from {table_name} where dt_int > '{table_time}' and dt_int <= '{e_time}';"""

        # sql = f"""select * from {table_name} where dt_int > '{table_time}' and dt_int <= '{e_time}';"""

        conn = self.pool.connection()

        try:
            df_db = pd.read_sql(sql, conn).drop(columns=['dt']).set_index('dt_int')
            conn.close()
            df_concat = pd.concat([df_par, df_db]).sort_values(by=['dt_int', 'symbol'])
            return df_concat

        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()

    def query_barra_returns(self, table_name, start_time, end_time):
        try:
            s_time = int(convert_to_datetime(start_time).strftime('%Y%m%d'))
            e_time = int(convert_to_datetime(end_time).strftime('%Y%m%d'))
        except Exception as e:
            print(f"Error, start time or end time is invalid, detail is {e}.")
            return pd.DataFrame()

        file_path = os.path.join(base_lib_path, 'return', table_name)
        file_name = os.listdir(file_path)[0]
        table_time = int(file_name.split('.')[0])
        table_path = os.path.join(file_path, file_name)
        if not os.path.exists(table_path):
            print('Error, table is not exists.')
            return pd.DataFrame()

        df_par = pd.read_parquet(table_path)
        if df_par.index[0] > s_time:
            print('Error, the start time is too early.')
            return pd.DataFrame()

        if e_time <= table_time:
            df_par = df_par.loc[(s_time <= df_par.index) & (e_time >= df_par.index)]
            return df_par

        df_par = df_par.loc[s_time:df_par.index[-1]]
        if table_time < s_time:
            sql = f"""select * from {table_name} where dt_int >= '{s_time}' and dt_int <= '{e_time}';"""
        else:
            sql = f"""select * from {table_name} where dt_int > '{table_time}' and dt_int <= '{e_time}';"""
        conn = self.pool.connection()
        try:
            df_db = pd.read_sql(sql, conn).drop(columns=['dt']).set_index('dt_int')
            conn.close()
            df_concat = pd.concat([df_par, df_db])
            return df_concat
        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()


# class BaseLib(object):
#     def __init__(self, mysql_info=BASE_LIB_MYSQL, logger=None):
#         self.logger = logger
#         self.pool = PooledDB(pymysql, 24, **mysql_info)
#
#     def get_tables(self):
#         sql = "show tables;"
#         conn = self.pool.connection()
#         curs = conn.cursor()
#
#         try:
#             curs.execute(sql)
#             res = curs.fetchall()
#             tables = [table[0] for table in res]
#
#         except Exception as e:
#             raise e
#
#         else:
#             return tables
#
#         finally:
#             conn.close()
#
#     def check_tables(self):
#         conn = self.pool.connection()
#         curs = conn.cursor()
#
#         create_table_info = """
#         CREATE TABLE `factor_exposure_hs300_hs300` (
#           `symbol` varchar(16) NOT NULL,
#           `dt` date NOT NULL,
#           `dt_int` int(8) NOT NULL,
#           `beta` double DEFAULT NULL,
#           `residual_volatility` double DEFAULT NULL,
#           `earnings_variability` double DEFAULT NULL,
#           `BP` double DEFAULT NULL,
#           `earnings_quality` double DEFAULT NULL,
#           `growth` double DEFAULT NULL,
#           `investment_quality` double DEFAULT NULL,
#           `leverage` double DEFAULT NULL,
#           `liquidity` double DEFAULT NULL,
#           `momentum` double DEFAULT NULL,
#           `profitability` double DEFAULT NULL,
#           `long_term_reversal` double DEFAULT NULL,
#           `short_term_reversal` double DEFAULT NULL,
#           `size` double DEFAULT NULL,
#           `mid_cap` double DEFAULT NULL,
#           `earnings_yield` double DEFAULT NULL,
#           `create_date` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
#           PRIMARY KEY (`symbol`,`dt`),
#           KEY `idx_0` (`dt_int`)
#         ) ENGINE=InnoDB DEFAULT CHARSET=utf8
#         """
#         try:
#             curs.execute(create_table_info)
#
#         except Exception as e:
#             # print(f"""{datetime.datetime.now()}, create table error, error is {e}.""")
#             if self.logger is not None:
#                 self.logger.error(f"""{datetime.datetime.now()}, create table error, error is {e}.""")
#             conn.rollback()
#         else:
#             conn.commit()
#
#         conn.close()
#
#     def generate_cache_data(self):
#         conn = self.pool.connection()
#         tables = self.get_tables()
#         return_tables = [
#             "factor_returns_hs300",
#             "factor_returns_investable",
#             "factor_returns_zz500"
#         ]
#
#         factor_tables = [
#             "factor_exposure_hs300_hs300",
#             "factor_exposure_investable_zz800",
#             "factor_exposure_zz500_zz500",
#         ]
#
#         for table in tables:
#             sql = f"""select * from {table};"""
#             df = pd.read_sql(sql, conn).set_index('dt_int').sort_index()
#
#             if table in return_tables:
#                 path = os.path.join(base_path, 'return', table)
#
#             elif table in factor_tables:
#                 path = os.path.join(base_path, 'factor', table)
#
#             else:
#                 print(f'Error, {table} not in return tables or factor tables.')
#                 return
#
#             last_time = df.index[-1]
#             str_last_time = str(last_time)
#
#             if not os.path.exists(path):
#                 os.makedirs(path)
#
#             if os.listdir(path):
#                 os.system(f'cd {path} && rm *.par')
#
#             # dt_int index
#             df.drop(columns=['dt'], inplace=True)
#             print(f'{path}/{str_last_time}.par')
#             df.to_parquet(f'{path}/{str_last_time}.par')
#
#         conn.close()
#
#     def query_barra_factors(self, table_name, start_time, end_time):
#         try:
#             s_time = int(convert_to_datetime(start_time).strftime('%Y%m%d'))
#             e_time = int(convert_to_datetime(end_time).strftime('%Y%m%d'))
#
#         except Exception as e:
#             print(f"Error, start time or end time is invalid, detail is {e}.")
#             return pd.DataFrame()
#
#         file_path = os.path.join(base_path, 'factor', table_name)
#         file_name = os.listdir(file_path)[0]
#         table_time = int(file_name.split('.')[0])
#
#         table_path = os.path.join(file_path, file_name)
#
#         if not os.path.exists(table_path):
#             print('Error, table is not exists.')
#             return pd.DataFrame()
#
#         df_par = pd.read_parquet(table_path)
#
#         if df_par.index[0] > s_time:
#             print('Error, the start time is too early.')
#             return pd.DataFrame()
#
#         if e_time <= table_time:
#             df_par = df_par.loc[(s_time <= df_par.index) & (e_time >= df_par.index)]
#             return df_par
#
#         df_par = df_par.loc[s_time <= df_par.index]
#
#         sql = f"""select * from {table_name} where dt_int > '{table_time}' and dt_int <= '{e_time}';"""
#
#         conn = self.pool.connection()
#
#         try:
#             df_db = pd.read_sql(sql, conn).drop(columns=['dt']).set_index('dt_int')
#             conn.close()
#             df_concat = pd.concat([df_par, df_db])
#             return df_concat
#
#         except Exception as e:
#             print(f'Error, details is {e}.')
#             return pd.DataFrame()
#
#     def query_barra_returns(self, table_name, start_time, end_time):
#         try:
#             s_time = int(convert_to_datetime(start_time).strftime('%Y%m%d'))
#             e_time = int(convert_to_datetime(end_time).strftime('%Y%m%d'))
#
#         except Exception as e:
#             print(f"Error, start time or end time is invalid, detail is {e}.")
#             return pd.DataFrame()
#
#         file_path = os.path.join(base_path, 'return', table_name)
#         file_name = os.listdir(file_path)[0]
#         table_time = int(file_name.split('.')[0])
#
#         table_path = os.path.join(file_path, file_name)
#
#         if not os.path.exists(table_path):
#             print('Error, table is not exists.')
#             return pd.DataFrame()
#
#         df_par = pd.read_parquet(table_path)
#
#         if df_par.index[0] > s_time:
#             print('Error, the start time is too early.')
#             return pd.DataFrame()
#
#         if e_time <= table_time:
#             df_par = df_par.loc[(s_time <= df_par.index) & (e_time >= df_par.index)]
#             return df_par
#
#         df_par = df_par[s_time <= df_par.index]
#         sql = f"""select * from {table_name} where dt_int > '{table_time}' and dt_int <= '{e_time}';"""
#         conn = self.pool.connection()
#         try:
#             df_db = pd.read_sql(sql, conn).drop(columns=['dt']).set_index('dt_int')
#             conn.close()
#             df_concat = pd.concat([df_par, df_db])
#             return df_concat
#         except Exception as e:
#             print(f'Error, details is {e}.')
#             return pd.DataFrame()
#
#     def insert_factor_exposure(self, conn, hs300: pd.DataFrame, zz800: pd.DataFrame, zz500: pd.DataFrame,
#                                type="exposure") -> bool:
#         curs = conn.cursor()
#
#         def to_mysql(data, table):
#             dbItem = data
#             params = []
#             for key in dbItem:
#                 if str(dbItem[key]) == "nan":
#                     dbItem[key] = None
#                 params.append(dbItem[key])
#             sql = f"INSERT INTO {table} ({','.join(dbItem.keys())}) VALUES ({','.join(['%s' for i in range(len(dbItem.keys()))])}) "
#
#             sql += "ON DUPLICATE KEY UPDATE create_date=%s,"
#             params.append(DateTime.Now().ToString())
#             for key in dbItem:
#                 if key == "dt":
#                     continue
#                 if key == "symbol":
#                     continue
#                 if dbItem[key]:
#                     sql += key + "=%s,"
#                     params.append(dbItem[key])
#             sql = sql[:-1] + ";"
#             curs.execute(sql, params)
#             conn.commit()
#
#         def insert_df(df, table):
#             for index, item in df.iterrows():
#                 dbItem = item.to_dict()
#                 if type == "exposure":
#                     dbItem["dt"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyy-MM-dd")
#                     dbItem["dt_int"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyyMMdd")
#                     dbItem["symbol"] = index
#                     dbItem["symbol"] = str(dbItem["symbol"]).zfill(6)
#                     del dbItem["date"]
#                 elif type == "return":
#                     dbItem["dt"] = DateTime.AutoConvert(str(index)).ToString("yyyy-MM-dd")
#                     dbItem["dt_int"] = DateTime.AutoConvert(str(index)).ToString("yyyyMMdd")
#                 to_mysql(dbItem, table)
#
#         if type == "exposure":
#             insert_df(hs300, "factor_exposure_hs300_hs300")
#             insert_df(zz800, "factor_exposure_investable_zz800")
#             insert_df(zz500, "factor_exposure_zz500_zz500")
#             return True
#         elif type == "return":
#             insert_df(hs300, "factor_returns_hs300")
#             insert_df(zz800, "factor_returns_investable")
#             insert_df(zz500, "factor_returns_zz500")
#             return True
#         else:
#             return False
#
#     def insert_factor_exposure_new(self, conn, df_all: pd.DataFrame, type="exposure") -> bool:
#         curs = conn.cursor()
#
#         def to_mysql(data, table):
#             dbItem = data
#             params = []
#             for key in dbItem:
#                 if str(dbItem[key]) == "nan":
#                     dbItem[key] = None
#                 params.append(dbItem[key])
#             sql = f"INSERT INTO {table} ({','.join(dbItem.keys())}) VALUES ({','.join(['%s' for i in range(len(dbItem.keys()))])}) "
#
#             sql += "ON DUPLICATE KEY UPDATE create_date=%s,"
#             params.append(DateTime.Now().ToString())
#             for key in dbItem:
#                 if key == "dt":
#                     continue
#                 if key == "symbol":
#                     continue
#                 if dbItem[key]:
#                     sql += key + "=%s,"
#                     params.append(dbItem[key])
#             sql = sql[:-1] + ";"
#             curs.execute(sql, params)
#             conn.commit()
#
#         def insert_df(df, table):
#             for index, item in df.iterrows():
#                 dbItem = item.to_dict()
#                 if type == "exposure":
#                     dbItem["dt"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyy-MM-dd")
#                     dbItem["dt_int"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyyMMdd")
#                     dbItem["symbol"] = index
#                     dbItem["symbol"] = str(dbItem["symbol"]).zfill(6)
#                     del dbItem["date"]
#                 elif type == "return":
#                     dbItem["dt"] = DateTime.AutoConvert(str(index)).ToString("yyyy-MM-dd")
#                     dbItem["dt_int"] = DateTime.AutoConvert(str(index)).ToString("yyyyMMdd")
#                 to_mysql(dbItem, table)
#
#         if type == "exposure":
#             insert_df(df_all, "factor_exposure_all_wdqa")
#             return True
#         elif type == "return":
#             insert_df(hs300, "factor_returns_hs300")
#             insert_df(zz800, "factor_returns_investable")
#             insert_df(zz500, "factor_returns_zz500")
#             return True
#         else:
#             return False
