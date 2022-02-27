import datetime
import os

import numpy as np
import pandas as pd
import pymysql
from ExObject.DateTime import DateTime
from Platform_ops.config.config import configs, TEST_FACTORS_DAILY_TI0
from Research.database.mysql import MysqlAPI, BaseLib
from DevTools.tools.ding import Ding
import traceback

LIB_PATH, EVAL_UNIVERSES, EVAL_IC_PRICES, EVAL_TRADING_PERIODS, EVAL_RETURN_PRICES, SQL_BATCH_SIZE = \
    configs.LIB_PATH, configs.EVAL_UNIVERSES, configs.EVAL_IC_PRICES, configs.EVAL_TRADING_PERIODS, \
    configs.EVAL_RETURN_PRICES, int(configs.SQL_BATCH_SIZE)
ding = Ding(configs.DING_SECRET_KEY, configs.DING_ACCESS_TOKEN)
factor_lib_path = os.path.join(LIB_PATH, 'factor_lib_path')
base_lib_path = os.path.join(LIB_PATH, 'base_lib')


class MysqlOps(MysqlAPI):
    def __init__(self, mysql_info, logger=None):
        super().__init__(mysql_info=mysql_info, threads=4, logger=logger)

    def _create_alpha_table(self, table_name):
        conn = self.pool.connection()
        curs = conn.cursor()

        try:
            create_sql = f"""
                create table if not exists {table_name}
                (
                    id                     int auto_increment                  primary key,
                    symbol                 varchar(16)                         not null,
                    trading_time          datetime                            not null,
                    alpha_value            double                              not null,
                    create_time            timestamp default CURRENT_TIMESTAMP not null,
                    constraint trading_time
                        unique (trading_time, symbol)
                );
            """

            curs.execute(create_sql)

        except Exception as e:
            if self.logger is not None:
                self.logger.error(e)
            conn.rollback()
            return False
        else:
            conn.commit()

        conn.close()
        return True

    # 2. batch insert dataframe to db
    def _insert_data(self, table_name, key_sql, value_sql, values, conn):
        curs = conn.cursor()

        try:
            insert_data_str = """insert into %s (%s) values (%s)""" % (table_name, key_sql, value_sql)
            curs.executemany(insert_data_str, values)

        except Exception as e:
            ding.send_ding("ERROR | MysqlOps",
                           f"Error | Execute sql with table {table_name} error | Err-msg: {traceback.format_exc()}")
            if self.logger:
                self.logger.error(f"execute sql with table {table_name} error, error is {e}.")
            conn.rollback()
            return False
        else:
            conn.commit()

        return True

    def _replace_data(self, table_name, key_sql, value_sql, values, conn):
        curs = conn.cursor()

        try:
            insert_data_str = """replace into %s (%s) values (%s)""" % (table_name, key_sql, value_sql)
            curs.executemany(insert_data_str, values)

        except Exception as e:
            ding.send_ding("ERROR | MysqlOps",
                           f"Error | Execute sql with table {table_name} error | Err-msg: {traceback.format_exc()}")
            if self.logger:
                self.logger.error(f"execute sql with table {table_name} error, error is {e}.")
            conn.rollback()
            return False
        else:
            conn.commit()

        return True

    def insert_to_db(self, dataframe: pd.DataFrame, table_name):
        keys = dataframe.keys()
        conn = self.pool.connection()

        cols = ['symbol', 'trading_time', 'alpha_value']

        for col in cols:
            if col not in keys:
                print(f"Error! Column '{col}' must be exists.")
                return

        dataframe['alpha_value'] = np.where(np.isinf(dataframe['alpha_value']), np.nan, dataframe['alpha_value'])
        dataframe = dataframe.where(dataframe.notnull(), None)

        status_code = self._create_alpha_table(table_name)

        if not status_code:
            return

        length = len(dataframe)

        # start numbers
        s_num = 0

        # end numbers
        e_num = SQL_BATCH_SIZE

        # increment numbers (add)
        a_num = SQL_BATCH_SIZE

        key_sql = ','.join(keys)
        value_sql = ','.join(['%s'] * dataframe.shape[1])
        dataframe = dataframe.sort_values('trading_time')

        nums = 0
        while True:
            if e_num > length:
                child_df = dataframe.iloc[s_num:length]
                values = child_df.values.tolist()
                flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
                # flag = self._replace_data(table_name, key_sql, value_sql, values, conn)
                if flag:
                    nums += len(child_df)
                break

            child_df = dataframe.iloc[s_num:e_num]
            values = child_df.values.tolist()
            flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
            if flag:
                nums += len(child_df)

            s_num += a_num
            e_num += a_num

        if self.logger is not None:
            self.logger.info(f"""Success insert {nums} rows of data, total is {length}, failed is {length - nums}.""")

        conn.close()

    def check_table(self, return_table=False, ic_table=False):
        conn = self.pool.connection()
        curs = conn.cursor()

        create_alpha_info_enrollment = f"""
            create table if not exists alpha_info_enrollment
            (
                id                 int auto_increment primary key,
                researcher         varchar(24)                           not null,
                alpha_name         varchar(96) unique                    not null,
                status             varchar(24) default 'accepted'        not null,
                trade_direction    int                                   not null,
                category           varchar(16)                       not null,
                cover_rate     float                                 not null,
                critical_value     float                                 null,
                create_time        timestamp   default CURRENT_TIMESTAMP not null,
                update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                change_reason      text                                  null
            ) ENGINE = INNODB
              DEFAULT CHARSET = utf8mb4;
                """

        create_alpha_info_rejected = f"""
            create table if not exists alpha_info_rejected
            (
                id                 int auto_increment primary key,
                researcher         varchar(24)                           not null,
                alpha_name         varchar(96) unique                    not null,
                category           varchar(16)    not null,
                status             varchar(24) default 'rejected'        not null,
                cover_rate     float                                 not null,
                update_time   timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                change_reason      text                                  null
            ) ENGINE = INNODB
              DEFAULT CHARSET = utf8mb4;
                """

        create_alpha_data_info = f"""
                 create table if not exists alpha_data_info
                 (
                     alpha_name                 varchar(96)         primary key,
                     start_time                datetime                not null,
                     end_time                datetime               not null,
                    update_time   timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                 ) ENGINE = INNODB
                   DEFAULT CHARSET = utf8mb4;
                     """

        create_alpha_ops_info = f"""
                      create table if not exists alpha_data_info
                      (
                        alpha_name                 varchar(96)         primary key,
                        modules                   varchar(50)    not null,
                        update_time   timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                      ) ENGINE = INNODB
                        DEFAULT CHARSET = utf8mb4;
                          """

        try:
            curs.execute(create_alpha_info_enrollment)
            curs.execute(create_alpha_info_rejected)
            curs.execute(create_alpha_data_info)
            curs.execute(create_alpha_ops_info)

            if ic_table:
                create_recent_ic_table = """
                    CREATE TABLE if not exists alpha_recent_icir (
                        alpha_name varchar(96) NOT NULL,
                        universe varchar(32) NOT NULL,
                        period_in_day varchar(8) NOT NULL,
                        rec_1m_IC float DEFAULT NULL NULL,
                        rec_1m_IR float DEFAULT NULL NULL,
                        rec_3m_IC float DEFAULT NULL NULL,
                        rec_3m_IR float DEFAULT NULL NULL,
                        rec_6m_IC float DEFAULT NULL NULL,
                        rec_6m_IR float DEFAULT NULL NULL,
                        rec_1y_IC float DEFAULT NULL NULL,
                        rec_1y_IR float DEFAULT NULL NULL,
                        rec_2y_IC float DEFAULT NULL NULL,
                        rec_2y_IR float DEFAULT NULL NULL,
                        rec_3y_IC float DEFAULT Null NULL,
                        rec_3y_IR float DEFAULT NULL NULL,
                        update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (`alpha_name`, `universe`, `period_in_day`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                curs.execute(create_recent_ic_table)

                for benchmark in EVAL_UNIVERSES:
                    for bt_price in EVAL_IC_PRICES:
                        if bt_price in ('twap', 'vwap'):
                            for tp in EVAL_TRADING_PERIODS:
                                create_alpha_ic_daily_close = f"""
                                    CREATE TABLE if not exists alpha_daily_ic_{benchmark}_{bt_price}_{tp} (
                                        alpha_name varchar(96) NOT NULL,
                                        trading_time datetime NOT NULL,
                                        IC_1min float NOT NULL,
                                        IC_5min float NOT NULL,
                                        IC_15min float NOT NULL,
                                        IC_30min float NOT NULL,
                                        IC_60min float NOT NULL,
                                        IC_120min float NOT NULL,
                                        IC_1d float NOT NULL,
                                        IC_2d float NOT NULL,
                                        IC_3d float NOT NULL,
                                        IC_4d float NOT NULL,
                                        IC_5d float NOT NULL,
                                        IC_10d float NOT NULL,
                                        IC_20d float NOT NULL,
                                        create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                        PRIMARY KEY (`alpha_name`, `trading_time`)
                                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                                curs.execute(create_alpha_ic_daily_close)
                        else:
                            create_alpha_ic_daily_close = f"""
                                CREATE TABLE if not exists alpha_daily_ic_{benchmark}_{bt_price}_0 (
                                    alpha_name varchar(96) NOT NULL,
                                    trading_time datetime NOT NULL,
                                    IC_1min float NOT NULL,
                                    IC_5min float NOT NULL,
                                    IC_15min float NOT NULL,
                                    IC_30min float NOT NULL,
                                    IC_60min float NOT NULL,
                                    IC_120min float NOT NULL,
                                    IC_1d float NOT NULL,
                                    IC_2d float NOT NULL,
                                    IC_3d float NOT NULL,
                                    IC_4d float NOT NULL,
                                    IC_5d float NOT NULL,
                                    IC_10d float NOT NULL,
                                    IC_20d float NOT NULL,
                                    create_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                    PRIMARY KEY (`alpha_name`, `trading_time`)
                                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                            curs.execute(create_alpha_ic_daily_close)

            if return_table:
                create_recent_return_table = """
                    CREATE TABLE if not exists alpha_recent_returns (
                        alpha_name varchar(96) NOT NULL,
                        universe varchar(32) NOT NULL,
                        rec_1m_long_ret float DEFAULT NULL,
                        rec_1m_short_ret float DEFAULT NULL,
                        rec_1m_ls_ret float DEFAULT NULL,
                        rec_1m_long_turnover float DEFAULT NULL,
                        rec_1m_short_turnover float DEFAULT NULL,
                        rec_1m_ls_turnover float DEFAULT NULL,
                        rec_1m_long_sharpe float DEFAULT NULL,
                        rec_1m_short_sharpe float DEFAULT NULL,
                        rec_1m_ls_sharpe float DEFAULT NULL,
                        rec_3m_long_ret float DEFAULT NULL,
                        rec_3m_short_ret float DEFAULT NULL,
                        rec_3m_ls_ret float DEFAULT NULL,
                        rec_3m_long_turnover float DEFAULT NULL,
                        rec_3m_short_turnover float DEFAULT NULL,
                        rec_3m_ls_turnover float DEFAULT NULL,
                        rec_3m_long_sharpe float DEFAULT NULL,
                        rec_3m_short_sharpe float DEFAULT NULL,
                        rec_3m_ls_sharpe float DEFAULT NULL,
                        rec_6m_long_ret float DEFAULT NULL,
                        rec_6m_short_ret float DEFAULT NULL,
                        rec_6m_ls_ret float DEFAULT NULL,
                        rec_6m_long_turnover float DEFAULT NULL,
                        rec_6m_short_turnover float DEFAULT NULL,
                        rec_6m_ls_turnover float DEFAULT NULL,
                        rec_6m_long_sharpe float DEFAULT NULL,
                        rec_6m_short_sharpe float DEFAULT NULL,
                        rec_6m_ls_sharpe float DEFAULT NULL,
                        rec_1y_long_ret float DEFAULT NULL,
                        rec_1y_short_ret float DEFAULT NULL,
                        rec_1y_ls_ret float DEFAULT NULL,
                        rec_1y_long_turnover float DEFAULT NULL,
                        rec_1y_short_turnover float DEFAULT NULL,
                        rec_1y_ls_turnover float DEFAULT NULL,
                        rec_1y_long_sharpe float DEFAULT NULL,
                        rec_1y_short_sharpe float DEFAULT NULL,
                        rec_1y_ls_sharpe float DEFAULT NULL,
                        rec_2y_long_ret float DEFAULT NULL,
                        rec_2y_short_ret float DEFAULT NULL,
                        rec_2y_ls_ret float DEFAULT NULL,
                        rec_2y_long_turnover float DEFAULT NULL,
                        rec_2y_short_turnover float DEFAULT NULL,
                        rec_2y_ls_turnover float DEFAULT NULL,
                        rec_2y_long_sharpe float DEFAULT NULL,
                        rec_2y_short_sharpe float DEFAULT NULL,
                        rec_2y_ls_sharpe float DEFAULT NULL,
                        rec_3y_long_ret float DEFAULT NULL,
                        rec_3y_short_ret float DEFAULT NULL,
                        rec_3y_ls_ret float DEFAULT NULL,
                        rec_3y_long_turnover float DEFAULT NULL,
                        rec_3y_short_turnover float DEFAULT NULL,
                        rec_3y_ls_turnover float DEFAULT NULL,
                        rec_3y_long_sharpe float DEFAULT NULL,
                        rec_3y_short_sharpe float DEFAULT NULL,
                        rec_3y_ls_sharpe float DEFAULT NULL,
                        update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (`alpha_name`, `universe`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                curs.execute(create_recent_return_table)

                create_recent_return_after_cost_table = """
                    CREATE TABLE if not exists alpha_recent_return_after_cost (
                        alpha_name varchar(96) NOT NULL,
                        universe varchar(32) NOT NULL,
                        rec_1m_long_ret_afc_1 float DEFAULT NULL,
                        rec_1m_long_ret_afc_2 float DEFAULT NULL,
                        rec_1m_short_ret_afc_1 float DEFAULT NULL,
                        rec_1m_short_ret_afc_2 float DEFAULT NULL,
                        rec_1m_ls_ret_afc_1 float DEFAULT NULL,
                        rec_1m_ls_ret_afc_2 float DEFAULT NULL,
                        rec_3m_long_ret_afc_1 float DEFAULT NULL,
                        rec_3m_long_ret_afc_2 float DEFAULT NULL,
                        rec_3m_short_ret_afc_1 float DEFAULT NULL,
                        rec_3m_short_ret_afc_2 float DEFAULT NULL,
                        rec_3m_ls_ret_afc_1 float DEFAULT NULL,
                        rec_3m_ls_ret_afc_2 float DEFAULT NULL,
                        rec_6m_long_ret_afc_1 float DEFAULT NULL,
                        rec_6m_long_ret_afc_2 float DEFAULT NULL,
                        rec_6m_short_ret_afc_1 float DEFAULT NULL,
                        rec_6m_short_ret_afc_2 float DEFAULT NULL,
                        rec_6m_ls_ret_afc_1 float DEFAULT NULL,
                        rec_6m_ls_ret_afc_2 float DEFAULT NULL,
                        rec_1y_long_ret_afc_1 float DEFAULT NULL,
                        rec_1y_long_ret_afc_2 float DEFAULT NULL,
                        rec_1y_short_ret_afc_1 float DEFAULT NULL,
                        rec_1y_short_ret_afc_2 float DEFAULT NULL,
                        rec_1y_ls_ret_afc_1 float DEFAULT NULL,
                        rec_1y_ls_ret_afc_2 float DEFAULT NULL,
                        rec_2y_long_ret_afc_1 float DEFAULT NULL,
                        rec_2y_long_ret_afc_2 float DEFAULT NULL,
                        rec_2y_short_ret_afc_1 float DEFAULT NULL,
                        rec_2y_short_ret_afc_2 float DEFAULT NULL,
                        rec_2y_ls_ret_afc_1 float DEFAULT NULL,
                        rec_2y_ls_ret_afc_2 float DEFAULT NULL,
                        rec_3y_long_ret_afc_1 float DEFAULT NULL,
                        rec_3y_long_ret_afc_2 float DEFAULT NULL,
                        rec_3y_short_ret_afc_1 float DEFAULT NULL,
                        rec_3y_short_ret_afc_2 float DEFAULT NULL,
                        rec_3y_ls_ret_afc_1 float DEFAULT NULL,
                        rec_3y_ls_ret_afc_2 float DEFAULT NULL,
                        update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        PRIMARY KEY (`alpha_name`, `universe`)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                curs.execute(create_recent_return_after_cost_table)

                for benchmark in EVAL_UNIVERSES:
                    for bt_price in EVAL_RETURN_PRICES:
                        if bt_price in ('twap', 'vwap'):
                            for tp in EVAL_TRADING_PERIODS:
                                create_factors_return_daily_group5_close = f"""
                                    CREATE TABLE if not exists alpha_daily_return_{benchmark}_{bt_price}_{tp}_group5 (
                                        alpha_name varchar(96) NOT NULL,
                                        trading_time DateTime NOT NULL,
                                        alpha_group0 float DEFAULT NULL,
                                        alpha_group1 float DEFAULT NULL,
                                        alpha_group2 float DEFAULT NULL,
                                        alpha_group3 float DEFAULT NULL,
                                        alpha_group4 float DEFAULT NULL,
                                        net_group0 float DEFAULT NULL,
                                        net_group1 float DEFAULT NULL,
                                        net_group2 float DEFAULT NULL,
                                        net_group3 float DEFAULT NULL,
                                        net_group4 float DEFAULT NULL,
                                        turnover_group0 float DEFAULT NULL,
                                        turnover_group1 float DEFAULT NULL,
                                        turnover_group2 float DEFAULT NULL,
                                        turnover_group3 float DEFAULT NULL,
                                        turnover_group4 float DEFAULT NULL,
                                        stock_num_group0 float DEFAULT NULL,
                                        stock_num_group1 float DEFAULT NULL,
                                        stock_num_group2 float DEFAULT NULL,
                                        stock_num_group3 float DEFAULT NULL,
                                        stock_num_group4 float DEFAULT NULL,
                                        benchmark_value float DEFAULT NULL,
                                        update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                        PRIMARY KEY (`alpha_name`,`trading_time`)
                                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                                curs.execute(create_factors_return_daily_group5_close)
                        else:
                            create_factors_return_daily_group5_close = f"""
                                CREATE TABLE if not exists alpha_daily_return_{benchmark}_{bt_price}_0_group5 (
                                    alpha_name varchar(96) NOT NULL,
                                    trading_time DateTime NOT NULL,
                                    alpha_group0 float DEFAULT NULL,
                                    alpha_group1 float DEFAULT NULL,
                                    alpha_group2 float DEFAULT NULL,
                                    alpha_group3 float DEFAULT NULL,
                                    alpha_group4 float DEFAULT NULL,
                                    net_group0 float DEFAULT NULL,
                                    net_group1 float DEFAULT NULL,
                                    net_group2 float DEFAULT NULL,
                                    net_group3 float DEFAULT NULL,
                                    net_group4 float DEFAULT NULL,
                                    turnover_group0 float DEFAULT NULL,
                                    turnover_group1 float DEFAULT NULL,
                                    turnover_group2 float DEFAULT NULL,
                                    turnover_group3 float DEFAULT NULL,
                                    turnover_group4 float DEFAULT NULL,
                                    stock_num_group0 float DEFAULT NULL,
                                    stock_num_group1 float DEFAULT NULL,
                                    stock_num_group2 float DEFAULT NULL,
                                    stock_num_group3 float DEFAULT NULL,
                                    stock_num_group4 float DEFAULT NULL,
                                    benchmark_value float DEFAULT NULL,
                                    update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                    PRIMARY KEY (`alpha_name`,`trading_time`)
                                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                            curs.execute(create_factors_return_daily_group5_close)

                    create_factors_return_daily_group5_holdings = f"""
                         CREATE TABLE if not exists alpha_daily_return_{benchmark}_holdings_group5 (
                             alpha_name varchar(96) NOT NULL,
                             date int not null,
                             symbol  varchar(16) not null,
                             alpha_value            double                              not null,
                             direction   varchar(8) not null,
                             create_time            timestamp default CURRENT_TIMESTAMP not null,
                             PRIMARY KEY  (`alpha_name`,`date`, `symbol`, `direction`)
                             ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                    curs.execute(create_factors_return_daily_group5_holdings)

                for benchmark in EVAL_UNIVERSES:
                    for bt_price in EVAL_RETURN_PRICES:
                        if bt_price in ('twap', 'vwap'):
                            for tp in EVAL_TRADING_PERIODS:
                                create_factors_return_daily_group10_close = f"""
                                    CREATE TABLE if not exists alpha_daily_return_{benchmark}_{bt_price}_{tp}_group10 (
                                        alpha_name varchar(96) NOT NULL,
                                        trading_time DateTime NOT NULL,
                                        alpha_group0 float DEFAULT NULL,
                                        alpha_group1 float DEFAULT NULL,
                                        alpha_group2 float DEFAULT NULL,
                                        alpha_group3 float DEFAULT NULL,
                                        alpha_group4 float DEFAULT NULL,
                                        alpha_group5 float DEFAULT NULL,
                                        alpha_group6 float DEFAULT NULL,
                                        alpha_group7 float DEFAULT NULL,
                                        alpha_group8 float DEFAULT NULL,
                                        alpha_group9 float DEFAULT NULL,
                                        net_group0 float DEFAULT NULL,
                                        net_group1 float DEFAULT NULL,
                                        net_group2 float DEFAULT NULL,
                                        net_group3 float DEFAULT NULL,
                                        net_group4 float DEFAULT NULL,
                                        net_group5 float DEFAULT NULL,
                                        net_group6 float DEFAULT NULL,
                                        net_group7 float DEFAULT NULL,
                                        net_group8 float DEFAULT NULL,
                                        net_group9 float DEFAULT NULL,
                                        turnover_group0 float DEFAULT NULL,
                                        turnover_group1 float DEFAULT NULL,
                                        turnover_group2 float DEFAULT NULL,
                                        turnover_group3 float DEFAULT NULL,
                                        turnover_group4 float DEFAULT NULL,
                                        turnover_group5 float DEFAULT NULL,
                                        turnover_group6 float DEFAULT NULL,
                                        turnover_group7 float DEFAULT NULL,
                                        turnover_group8 float DEFAULT NULL,
                                        turnover_group9 float DEFAULT NULL,
                                        stock_num_group0 float DEFAULT NULL,
                                        stock_num_group1 float DEFAULT NULL,
                                        stock_num_group2 float DEFAULT NULL,
                                        stock_num_group3 float DEFAULT NULL,
                                        stock_num_group4 float DEFAULT NULL,
                                        stock_num_group5 float DEFAULT NULL,
                                        stock_num_group6 float DEFAULT NULL,
                                        stock_num_group7 float DEFAULT NULL,
                                        stock_num_group8 float DEFAULT NULL,
                                        stock_num_group9 float DEFAULT NULL,
                                        benchmark_value float DEFAULT NULL,
                                        update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                        PRIMARY KEY (`alpha_name`,`trading_time`)
                                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                                curs.execute(create_factors_return_daily_group10_close)
                        else:
                            create_factors_return_daily_group10_close = f"""
                                CREATE TABLE if not exists alpha_daily_return_{benchmark}_{bt_price}_0_group10 (
                                    alpha_name varchar(96) NOT NULL,
                                    trading_time DateTime NOT NULL,
                                    alpha_group0 float DEFAULT NULL,
                                    alpha_group1 float DEFAULT NULL,
                                    alpha_group2 float DEFAULT NULL,
                                    alpha_group3 float DEFAULT NULL,
                                    alpha_group4 float DEFAULT NULL,
                                    alpha_group5 float DEFAULT NULL,
                                    alpha_group6 float DEFAULT NULL,
                                    alpha_group7 float DEFAULT NULL,
                                    alpha_group8 float DEFAULT NULL,
                                    alpha_group9 float DEFAULT NULL,
                                    net_group0 float DEFAULT NULL,
                                    net_group1 float DEFAULT NULL,
                                    net_group2 float DEFAULT NULL,
                                    net_group3 float DEFAULT NULL,
                                    net_group4 float DEFAULT NULL,
                                    net_group5 float DEFAULT NULL,
                                    net_group6 float DEFAULT NULL,
                                    net_group7 float DEFAULT NULL,
                                    net_group8 float DEFAULT NULL,
                                    net_group9 float DEFAULT NULL,
                                    turnover_group0 float DEFAULT NULL,
                                    turnover_group1 float DEFAULT NULL,
                                    turnover_group2 float DEFAULT NULL,
                                    turnover_group3 float DEFAULT NULL,
                                    turnover_group4 float DEFAULT NULL,
                                    turnover_group5 float DEFAULT NULL,
                                    turnover_group6 float DEFAULT NULL,
                                    turnover_group7 float DEFAULT NULL,
                                    turnover_group8 float DEFAULT NULL,
                                    turnover_group9 float DEFAULT NULL,
                                    stock_num_group0 float DEFAULT NULL,
                                    stock_num_group1 float DEFAULT NULL,
                                    stock_num_group2 float DEFAULT NULL,
                                    stock_num_group3 float DEFAULT NULL,
                                    stock_num_group4 float DEFAULT NULL,
                                    stock_num_group5 float DEFAULT NULL,
                                    stock_num_group6 float DEFAULT NULL,
                                    stock_num_group7 float DEFAULT NULL,
                                    stock_num_group8 float DEFAULT NULL,
                                    stock_num_group9 float DEFAULT NULL,
                                    benchmark_value float DEFAULT NULL,
                                    update_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                    PRIMARY KEY (`alpha_name`,`trading_time`)
                                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                            curs.execute(create_factors_return_daily_group10_close)

                    create_factors_return_daily_group10_holdings = f"""
                         CREATE TABLE if not exists alpha_daily_return_{benchmark}_holdings_group10 (
                             alpha_name varchar(96) NOT NULL,
                             date int not null,
                             symbol  varchar(16) not null,
                             alpha_value            double                              not null,
                             direction   varchar(8) not null,
                             create_time            timestamp default CURRENT_TIMESTAMP not null,
                             PRIMARY KEY (`alpha_name`,`date`, `symbol`, `direction`)
                             ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"""
                    curs.execute(create_factors_return_daily_group10_holdings)

        except Exception as e:
            print(f"""{datetime.datetime.now()}, create table error, error is {e}.""")
            if self.logger is not None:
                self.logger.error(f"""{datetime.datetime.now()}, create table error, error is {e}.""")
            conn.rollback()
        else:
            conn.commit()

        conn.close()

    def get_factor_info(self):
        conn = self.pool.connection()
        sql = f"""select * from alpha_info_enrollment order by create_time asc;"""
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    def get_factor_rejected_info(self):
        conn = self.pool.connection()
        sql = f"""select * from talpha_info_rejected;"""
        df = pd.read_sql(sql, conn)
        conn.close()
        return df

    # 3. register info to table info
    def register(self, author, alpha_name, trade_direction, cover_rate_value, critical_value,
                 status='accepted', change_reason=None, category='PV'):
        conn = self.pool.connection()
        curs = conn.cursor()

        if critical_value is not None:
            if change_reason is None:
                sql = f"""
                insert into 
                    alpha_info_enrollment (researcher, alpha_name, category, trade_direction, cover_rate, critical_value,  status) 
                values 
                    ('{author}', '{alpha_name}', '{trade_direction}', '{category}', '{cover_rate_value}', '{critical_value}', '{category}', '{status}');
                """
            else:
                sql = f"""
                insert into 
                    alpha_info_enrollment (researcher, alpha_name, category, trade_direction, cover_rate, critical_value, status, 
                    change_reason) 
                values 
                    ('{author}', '{alpha_name}', '{trade_direction}', '{category}', '{cover_rate_value}', '{critical_value}', 
                    '{status}', '{change_reason}');
                """

        else:
            if change_reason is None:
                sql = f"""
                insert into 
                    alpha_info_enrollment (researcher, alpha_name, trade_direction, cover_rate, category, status) 
                values 
                    ('{author}', '{alpha_name}', '{trade_direction}', '{cover_rate_value}', '{category}', '{status}');
                """
            else:
                sql = f"""
                insert into 
                    alpha_info_enrollment (researcher, alpha_name, trade_direction, cover_rate, category, status, change_reason) 
                values 
                    ('{author}', '{alpha_name}', '{trade_direction}', '{cover_rate_value}', '{category}', 
                    '{status}', '{change_reason}');
                """

        try:
            curs.execute(sql)
        except Exception as e:
            # print(f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
            if self.logger:
                self.logger.error(f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
            conn.rollback()
            raise pymysql.err.OperationalError(
                f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
        else:
            conn.commit()

        conn.close()

    def register_data_info(self, alpha_name, df_input):
        df = df_input.reset_index()
        start_time = str(df['timestamp'].min())
        end_time = str(df['timestamp'].max())
        conn = self.pool.connection()
        curs = conn.cursor()
        sql = f"""
                replace into 
                    alpha_data_info (alpha_name, start_time, end_time) 
                values 
                    ('{alpha_name}', '{start_time}', '{end_time}');
                """
        try:
            curs.execute(sql)
        except Exception as e:
            # print(f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
            if self.logger:
                self.logger.error(f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
            conn.rollback()
            conn.close()
            raise pymysql.err.OperationalError(
                f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
        else:
            conn.commit()

        conn.close()

    def update_data_info(self, alpha_name, df_input):
        df = df_input.reset_index()
        end_time = str(df['timestamp'].max())
        conn = self.pool.connection()
        curs = conn.cursor()
        sql = f"""
                UPDATE alpha_data_info SET end_time = '{end_time}' WHERE alpha_name = '{alpha_name}';
                """
        try:
            curs.execute(sql)
        except Exception as e:
            # print(f"""{datetime.datetime.now()}, insert into {alpha_name} error, error is {e}.""")
            if self.logger:
                self.logger.error(f"""{datetime.datetime.now()}, update date info {alpha_name} error, error is {e}.""")
            conn.rollback()
            conn.close()
            raise pymysql.err.OperationalError(
                f"""{datetime.datetime.now()},  update date info {alpha_name} error, error is {e}.""")
        else:
            conn.commit()

        conn.close()

    # 3. register info to table info rejected
    def register_rejected(self, author, alpha_name, cover_rate_value, change_reason=None, category='PV'):
        conn = self.pool.connection()
        curs = conn.cursor()
        status = 'rejected'

        if change_reason is None:
            sql = f"""
            replace into 
                alpha_info_rejected (researcher, alpha_name, category, cover_rate, status) 
            values 
                ('{author}', '{alpha_name}', '{category}', '{cover_rate_value}', '{status}');
            """
        else:
            sql = f"""
            replace into 
                alpha_info_rejected (researcher, alpha_name, category, cover_rate, status, change_reason) 
            values 
                ('{author}', '{alpha_name}', '{category}', '{cover_rate_value}', '{status}', '{change_reason}');
            """

        try:
            curs.execute(sql)
        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"""{datetime.datetime.now()} | Table: `alpha_info_reject`, 
                    insert into {alpha_name} error, error is {e}.""")
            conn.rollback()
            raise pymysql.err.OperationalError(
                f"""{datetime.datetime.now()} | Table: `alpha_info_reject`, 
                insert into {alpha_name} error, error is {e}.""")
        else:
            conn.commit()

        conn.close()

    # 4. modify status when table changed
    def update_to_watch(self, table_name, change_reason):
        conn = self.pool.connection()
        curs = conn.cursor()

        if isinstance(table_name, str):
            sql = f"""update alpha_info_enrollment set status='watched', change_reason='{change_reason}' where alpha_name='{table_name}';"""

            try:
                curs.execute(sql)
            except Exception as e:
                # print(f"""{sql} error, error is {e}.""")
                if self.logger:
                    self.logger.error(f"""{sql} error, error is {e}.""")
                conn.rollback()
            else:
                conn.commit()

        elif isinstance(table_name, list):
            values = []
            sql = f"""update alpha_info_enrollment set status='watched', change_reason=(%s) where alpha_name=(%s);"""

            for i in range(len(table_name)):
                values.append((change_reason[i], table_name[i]))

            try:
                curs.executemany(sql, values)
            except Exception as e:
                # print(f"""{sql} error, error is {e}.""")
                if self.logger:
                    self.logger.error(f"""{sql} error, error is {e}.""")
                conn.rollback()
            else:
                conn.commit()
        else:
            print('Error, table_name must be str or list.')
            return

        curs.close()
        conn.close()

    def remove(self, table_name):
        conn = self.pool.connection()
        curs = conn.cursor()

        remove_sql = f"""update alpha_info_enrollment set status = 'removed' where table_name = '{table_name}';"""

        drop_sql = f"""drop table {table_name};"""

        try:
            curs.execute(remove_sql)
            curs.execute(drop_sql)

        except Exception as e:
            # print(f"""remove error, error is {e}.""")
            if self.logger is not None:
                self.logger.error(f"""remove error, error is {e}.""")
            conn.rollback()

        else:
            conn.commit()

        conn.close()

    def get_tables(self, status=None):
        conn = self.pool.connection()
        sql = 'select * from alpha_info_enrollment'
        try:
            df = pd.read_sql(sql, con=conn)
            result_dic = df.groupby('researcher')['alpha_name'].apply(list).to_dict()
            conn.close()
            return result_dic
        except Exception as e:
            print(e)

    def generate_cache_data(self):
        conn = self.pool.connection()
        result_dic = self.get_tables()

        for name, tablelist in result_dic.items():
            for table in tablelist:
                sql = f"""
                   SELECT symbol, trading_time, alpha_value, create_time FROM {table} 
                   ORDER BY trading_time ASC, alpha_value DESC;"""
                df = pd.read_sql(sql, conn)
                last_time = df['trading_time'].iloc[-1]

                df = df.set_index(['trading_time', 'symbol'])
                path = os.path.join(factor_lib_path, name, table)
                int_last_time = int(datetime.datetime.strftime(last_time, '%Y%m%d'))

                if not os.path.exists(path):
                    os.makedirs(path)

                if os.listdir(path):
                    os.system(f'cd {path} && rm *.par')
                print(f'{path}/{int_last_time}.par')
                df.to_parquet(f'{path}/{int_last_time}.par')

        conn.close()

    def update_to_db(self, dataframe: pd.DataFrame, table_name):
        keys = dataframe.keys()
        conn = self.pool.connection()
        cols = ['symbol', 'trading_time', 'alpha_value']

        for col in cols:
            if col not in keys:
                print(f"Error! Column '{col}' must be exists.")
                return

        dataframe = dataframe.replace({np.NAN: None})

        length = len(dataframe)

        # start numbers
        s_num = 0

        # end numbers
        e_num = SQL_BATCH_SIZE

        # increment numbers (add)
        a_num = SQL_BATCH_SIZE

        key_sql = ','.join(keys)
        value_sql = ','.join(['%s'] * dataframe.shape[1])
        dataframe = dataframe.sort_values('trading_time')
        data_update_time = dataframe['trading_time'].iloc[-1]

        nums = 0
        while True:
            if e_num > length:
                child_df = dataframe.iloc[s_num:length]
                values = child_df.values.tolist()
                flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
                if flag:
                    nums += len(child_df)
                break

            child_df = dataframe.iloc[s_num:e_num]
            values = child_df.values.tolist()
            flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
            if flag:
                nums += len(child_df)

            s_num += a_num
            e_num += a_num

        if self.logger is not None:
            self.logger.info(f"""Success insert {nums} rows of data, total is {length}, failed is {length - nums}.""")

        conn.close()

    def insert_to_alpha_related_table(self, table_names, dataframe_list):
        conn = self.pool.connection()
        assert len(table_names) == len(dataframe_list), "Length of table_names and length of dataframe_list mismatch!"
        for i in range(len(table_names)):
            table_name = table_names[i]
            dataframe = dataframe_list[i]
            keys = dataframe.keys()
            dataframe.dropna(how='all', inplace=True)
            dataframe = dataframe.replace({np.NAN: None})

            length = len(dataframe)

            # start numbers
            s_num = 0

            # end numbers
            e_num = SQL_BATCH_SIZE

            # increment numbers (add)
            a_num = SQL_BATCH_SIZE

            key_sql = ','.join(keys)
            value_sql = ','.join(['%s'] * dataframe.shape[1])

            nums = 0
            while True:
                if e_num > length:
                    child_df = dataframe.iloc[s_num:length]
                    values = child_df.values.tolist()
                    # flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
                    flag = self._replace_data(table_name, key_sql, value_sql, values, conn)
                    if flag:
                        nums += len(child_df)
                    break

                child_df = dataframe.iloc[s_num:e_num]
                values = child_df.values.tolist()
                # flag = self._insert_data(table_name, key_sql, value_sql, values, conn)
                flag = self._replace_data(table_name, key_sql, value_sql, values, conn)
                if flag:
                    nums += len(child_df)

                s_num += a_num
                e_num += a_num

        conn.close()

    def query_factor_ic(self, name, start_time, end_time, benchmark='Investable', price='close', tp=0, period=('1d',)):
        conn = self.pool.connection()
        query_list = [f"IC_{item}" for item in period]
        query_str = ','.join(query_list)
        sql = f"""
        select trading_time, {query_str} from alpha_daily_ic_{benchmark}_{price}_{tp} where alpha_name='{name}' and 
        trading_time >= '{str(start_time)}' and trading_time <= '{str(end_time)}';
        """
        try:
            df = pd.read_sql(sql, conn)
            conn.close()
            return df
        except Exception as e:
            print(f'Error, details is {e}.')
            return pd.DataFrame()

    def query_factor_return(self, name, start_time, end_time, benchmark='Investable', price='close', tp=0, group=10):
        conn = self.pool.connection()
        query_list = [f"alpha_group{item}" for item in range(group)]
        query_str = ','.join(query_list)
        sql = f"""
        select trading_time, {query_str} from alpha_daily_return_{benchmark}_{price}_{tp}_group{group} 
        where alpha_name='{name}' and  trading_time >= '{str(start_time)}' and trading_time <= '{str(end_time)}';
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
        _period = []
        for item in period:
            item.replace('in', "")
            _period.append(item)
        query_list = [f"rec_{item}_IC" for item in _period]
        query_str = ','.join(query_list)
        sql = f"select {query_str} from alpha_recent_icir where alpha_name='{name}' and universe='{benchmark}';"
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
        sql = f"select {query_str} from alpha_recent_return_after_cost " \
              f"where alpha_name='{name}' and universe='{benchmark}';"
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


class BaseLibOps(BaseLib):
    def __init__(self, logger=None):
        super().__init__(logger=logger)

    def generate_cache_data(self):
        conn = self.pool.connection()
        tables = self.get_tables()
        return_tables = [
            "factor_returns_hs300",
            "factor_returns_investable",
            "factor_returns_zz500"
        ]

        factor_tables = [
            "factor_exposure_hs300_hs300",
            "factor_exposure_investable_zz800",
            "factor_exposure_zz500_zz500",
        ]

        for table in tables:
            sql = f"""select * from {table};"""
            df = pd.read_sql(sql, conn).sort_values(by=['dt_int', 'symbol']).set_index('dt_int')
            if table in return_tables:
                path = os.path.join(base_lib_path, 'return', table)

            elif table in factor_tables:
                path = os.path.join(base_lib_path, 'factor', table)

            else:
                print(f'Error, {table} not in return tables or factor tables.')
                return

            last_time = df.index[-1]
            str_last_time = str(last_time)

            if not os.path.exists(path):
                os.makedirs(path)

            if os.listdir(path):
                os.system(f'cd {path} && rm *.par')

            # dt_int index
            df.drop(columns=['dt'], inplace=True)
            print(f'{path}/{str_last_time}.par')
            df.to_parquet(f'{path}/{str_last_time}.par')

        conn.close()

    def insert_factor_exposure(self, conn, hs300: pd.DataFrame, zz800: pd.DataFrame, zz500: pd.DataFrame,
                               type="exposure") -> bool:
        curs = conn.cursor()

        def to_mysql(data, table):
            dbItem = data
            params = []
            for key in dbItem:
                if str(dbItem[key]) == "nan":
                    dbItem[key] = None
                params.append(dbItem[key])
            sql = f"INSERT INTO {table} ({','.join(dbItem.keys())}) VALUES ({','.join(['%s' for i in range(len(dbItem.keys()))])}) "

            sql += "ON DUPLICATE KEY UPDATE create_date=%s,"
            params.append(DateTime.Now().ToString())
            for key in dbItem:
                if key == "dt":
                    continue
                if key == "symbol":
                    continue
                if dbItem[key]:
                    sql += key + "=%s,"
                    params.append(dbItem[key])
            sql = sql[:-1] + ";"
            curs.execute(sql, params)
            conn.commit()

        def insert_df(df, table):
            for index, item in df.iterrows():
                dbItem = item.to_dict()
                if type == "exposure":
                    dbItem["dt"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyy-MM-dd")
                    dbItem["dt_int"] = DateTime.AutoConvert(str(dbItem["date"])).ToString("yyyyMMdd")
                    dbItem["symbol"] = index
                    dbItem["symbol"] = str(dbItem["symbol"]).zfill(6)
                    del dbItem["date"]
                elif type == "return":
                    dbItem["dt"] = DateTime.AutoConvert(str(index)).ToString("yyyy-MM-dd")
                    dbItem["dt_int"] = DateTime.AutoConvert(str(index)).ToString("yyyyMMdd")
                to_mysql(dbItem, table)

        if type == "exposure":
            insert_df(hs300, "factor_exposure_hs300_hs300")
            insert_df(zz800, "factor_exposure_investable_zz800")
            insert_df(zz500, "factor_exposure_zz500_zz500")
            return True
        elif type == "return":
            insert_df(hs300, "factor_returns_hs300")
            insert_df(zz800, "factor_returns_investable")
            insert_df(zz500, "factor_returns_zz500")
            return True
        else:
            return False
