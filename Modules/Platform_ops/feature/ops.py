import datetime
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import time
import traceback
import uuid
from urllib import parse

import numpy as np
import pandas as pd
import pyarrow
from DataAPI import get_next_trading_day
from DevTools.tools.elk_logger import get_elk_logger
from ExObject.DateTime import DateTime
from ExObject.TimeSpan import TimeSpan
from Platform_ops.config.config import configs, TEST_FACTORS_DAILY_TI0
from Platform_ops.database.mysql import MysqlOps
from Research.backtest.bt import BTDaily, BTFeatureDaily
from Research.feature.ft import FeatureAnalysis
from Research.metrics.entrance_requirements import get_feature_status, is_correlation_accepted
from Research.metrics.risk_analysis import calc_sharpe_ratio
from Research.utils import normalization as norm
from redis import Redis, ConnectionPool
from tabulate import tabulate
from tqdm import tqdm


class Distribute:
    def __init__(self, options, logger=None):
        REDIS = "redis://:{}@172.16.4.3:6379/0".format(parse.quote('unp.7k(nvDf2Vghy'))
        self.QUEUE_KEY_PRODUCER_P = "ft_factorlib"
        self.QUEUE_KEY_PRODUCER_UPDATE_P = "ft_factorlib_update"
        self.QUEUE_KEY_CONSUMER_P = "ft_distribute:queue"
        self.RESULT_PREKEY_P = "ft_distribute:result:"
        self.QUEUE_KEY_PRODUCER_T = "ft_factorlib_test"
        self.QUEUE_KEY_CONSUMER_T = "ft_distribute_test:queue"
        self.RESULT_PREKEY_T = "ft_distribute_test:result:"
        self.pool = ConnectionPool.from_url(REDIS, decode_components=True)
        self.redis = Redis(connection_pool=self.pool)
        self.context = pyarrow.default_serialization_context()
        if logger is None:
            self.logger = get_elk_logger("platform_correlation_service_test", console=True)
        else:
            self.logger = logger

        self.corr_matrix = None
        self.corr_end_time = None
        self.corr_start_time = None

        self.configs = options
        self.corr_path = os.path.join(self.configs.corr_path, self.configs.db, "corr", "corr.par")

    def load_corr_matrix(self):
        try:
            self.corr_matrix = pd.read_parquet(self.corr_path)
            self.corr_end_time = self.corr_matrix.index.get_level_values(0).max()
            self.corr_start_time = self.corr_matrix.index.get_level_values(0).min()
        except FileNotFoundError:
            if not os.path.exists(os.path.dirname(self.corr_path)):
                os.makedirs(os.path.dirname(self.corr_path))

    def to_redis_upload_dataframe(self, df):
        df_bytes = self.context.serialize(df).to_buffer().to_pybytes()
        self.redis.lpush(self.QUEUE_KEY_CONSUMER_P, df_bytes)
        md5 = hashlib.md5(df_bytes).hexdigest()
        feature_name = df.columns[0]

        c = 0
        while c < 60 * (self.configs.CORR_COMPUTE_TIMEOUT * self.configs.NUM_COMPUTE_NODE + 0.5) * 10:
            key = self.RESULT_PREKEY_P + md5
            r = self.redis.get(key)
            if r:
                self.redis.delete(key)
                df = self.context.deserialize(r)
                self.logger.info(
                    f'Result for corr_matrix of {feature_name} has been received.')
                return md5, df
            c += 1
            time.sleep(0.1)
        else:
            self.redis.delete(key)
            self.logger.error(f'Corr_matrix of {feature_name} has not been received. Time out...')
            return md5, False

    def to_redis_feature_status(self, md5, status, time_out=3600 * 12):
        key = self.RESULT_PREKEY_P + md5 + "_2"
        self.redis.set(key, json.dumps(status), time_out)

    def set_uuid(self, uuid_string, value, time_out=3600 * 12):
        self.redis.set(uuid_string, value, time_out)

    def distribute_correlation_calculator(self):
        """DISTRIBUTE WORKER"""
        self.load_corr_matrix()
        last_save = DateTime.Now()
        self.redis.delete(self.QUEUE_KEY_CONSUMER_P)
        self.logger.info("=" * 125)
        self.logger.info("Calculate correlation service starts...")
        self.logger.info("=" * 125)

        while True:
            df_bytes = self.redis.brpop(self.QUEUE_KEY_CONSUMER_P, timeout=60 * self.configs.BRPOP_TIMEOUT_MINUTES)
            if df_bytes is not None and len(df_bytes) > 1:
                df = self.context.deserialize(df_bytes[1])
                feature_name = df.columns[0]
                md5 = hashlib.md5(df_bytes[1]).hexdigest()

                # logger
                if df is not None:
                    self.logger.info(
                        f'Feature data of {feature_name} has been received, md5: {md5}, length: {df.shape[0]}')
                else:
                    self.logger.info(
                        f'Feature data of {feature_name} has been received, md5: {md5}, length: 0')

                result = Distribute.get_feature_correlation(df, self.corr_matrix)
                df_bytes = self.context.serialize(result).to_buffer().to_pybytes()
                self.redis.set(self.RESULT_PREKEY_P + md5, df_bytes, 3600 * 12)
                self.logger.info(
                    f'Correlation result of {feature_name} has been sent.')

                c = 0
                while c < 60 * self.configs.STATUS_COMPUTE_TIMEOUT * 10:
                    key = self.RESULT_PREKEY_P + md5 + "_2"
                    r = self.redis.get(key)
                    if r:
                        self.redis.delete(key)
                        result = json.loads(json.loads(r.decode("utf-8")))
                        self.logger.info(
                            f'Operation for corr_matrix of {feature_name} has been received.')
                        break
                    c += 1
                    time.sleep(0.1)
                else:
                    self.redis.delete(key)
                    self.logger.error(f'Operation for corr_matrix of {feature_name} not received. Time out...')
                    continue

                if result.get("status") == 'accepted_no_replace':
                    self.append_corr_matrix(df)
                    self.logger.info(
                        f'Operation for corr_matrix of {feature_name} has done. status: accepted_no_replace.')
                elif result.get("status") == 'accepted_with_replace':
                    self.replace_corr_matrix(result["alpha_name"], df)
                    self.logger.info(
                        f'Operation for corr_matrix of {feature_name} has done. status: accepted_with_replace.')
                else:
                    self.logger.info(
                        f'No operation for corr_matrix of {feature_name}.')

            if (DateTime.Now() - last_save) > TimeSpan(minute=self.configs.SAVE_TIME_SPAN_MINUTES):
                if self.corr_matrix is not None:
                    path_output = self.corr_path
                    folder = os.path.dirname(path_output)
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                    self.corr_matrix.to_parquet(path_output)
                    last_save = DateTime.Now()
                    self.logger.info(
                        f'Corr_matrix has been updated into {self.corr_path}.')

    @staticmethod
    def get_feature_correlation(feature_data, corr_matrix):
        """
        :param
        method: {'pearson', 'kendall', 'spearman'} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays
        """
        method = 'spearman'
        if corr_matrix is None:
            return pd.DataFrame()
        feature_name = feature_data.columns[0]

        corr_dict = dict()
        stock_num_count = []
        if feature_name in corr_matrix.columns:
            return
        grouped_target = corr_matrix.groupby(level=0)
        grouped_feature = feature_data.groupby(level=0)
        for timestamp, group in grouped_feature:
            group = group.reset_index(level=0, drop=True)
            try:
                group_target = grouped_target.get_group(timestamp).reset_index(level=0, drop=True)
            except KeyError:
                continue

            temp = group.join(group_target, how='inner')

            corr_dict[timestamp] = temp
            stock_num_count.append(temp.shape[0])

        target_info = [(key, corr_dict[key]) for key in corr_dict.keys()]
        columns = temp.columns.tolist()

        stock_iter = iter(target_info)
        worker_num = mp.cpu_count()
        pool = mp.Pool(processes=worker_num)

        # ³õÊ¼»¯ÈÎÎñ
        result_list = [pool.apply_async(FeatureAnalysis._calculate_ic,
                                        args=(next(stock_iter), columns, None, method,))
                       for _ in range(min(worker_num, len(target_info)))]

        df_list = []
        flag = 1
        with tqdm(total=len(target_info), ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        df = result_list[out_index].get()
                        if df:
                            df_list.append(df)
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("Calculating correlations...")
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(
                                    pool.apply_async(FeatureAnalysis._calculate_ic,
                                                     args=(next(stock_iter), columns, None, method,)))
                            except StopIteration:
                                flag = 0

        column_final = ['timestamp']
        for item in columns[1:]:
            column_final.append(item)
        corr_table = pd.DataFrame(df_list, columns=column_final)
        corr_table_result = corr_table.set_index('timestamp').mean().to_frame('correlation')
        corr_table_result.index.name = 'alpha_name'
        pool.terminate()
        return corr_table_result

    def append_corr_matrix(self, df):
        if self.corr_matrix is None:
            self.corr_end_time = df.index.get_level_values(0).max()
            self.corr_start_time = \
                pd.Timestamp(self.corr_end_time.year, self.corr_end_time.month, self.corr_end_time.day) - \
                pd.Timedelta(days=self.configs.CORR_PERIOD)
            self.corr_matrix = df.loc[
                (df.index.get_level_values(0) >= self.corr_start_time) &
                (df.index.get_level_values(0) <= self.corr_end_time)].sort_index()
        else:
            self.corr_matrix = self.corr_matrix.join(df.loc[
                                                         (df.index.get_level_values(
                                                             0) >= self.corr_start_time) &
                                                         (df.index.get_level_values(
                                                             0) <= self.corr_end_time)], how='outer')

    def replace_corr_matrix(self, alpha_name, df):
        self.corr_matrix = self.corr_matrix.drop(columns=alpha_name)
        self.corr_matrix = self.corr_matrix.join(
            df.loc[(df.index.get_level_values(0) <= self.corr_end_time) &
                   (df.index.get_level_values(0) >= self.corr_start_time)], how='outer')

    def add_all_factors_to_redis_for_file(self, mysql_info, check_factors_in_lib=True, researchers=None):
        """
        For files upload. Push alpha names into redis
        """
        fa = FeatureAnalysis(self.configs, mysql_info=mysql_info, feature_path=self.configs.FACTORS_PATH, logger=True)
        if fa.logger is not None:
            fa.logger.info("=" * 125)
            fa.logger.info(f"Database: {fa.options.db} has {len(fa.features_in_db)} factors")

        self.redis.delete(self.QUEUE_KEY_PRODUCER_P)
        tested_list = []

        if check_factors_in_lib:
            tested_list.extend(fa.features_in_db)

        count = 0
        for ft_name, uname in fa.features_in_path.items():
            if check_factors_in_lib:
                if ft_name in tested_list:
                    continue
            if researchers is None:
                dbItem = {
                    "uname": uname,
                    "ft_name": ft_name
                }
                count += 1
                self.redis.lpush(self.QUEUE_KEY_PRODUCER_P, json.dumps(dbItem))
            else:
                if uname in researchers:
                    dbItem = {
                        "uname": uname,
                        "ft_name": ft_name
                    }
                    count += 1
                    self.redis.lpush(self.QUEUE_KEY_PRODUCER_P, json.dumps(dbItem))

        self.logger.info("=" * 125)
        self.logger.info(
            f'Add all factors to redis queue, count: {count}.')
        return True

    def add_factors_to_redis_for_dataframe(self, researcher, category, dataframe_list, file_path, job_id, project_name,
                                           branch_name, clear_queue=False,
                                           time_out=3600 * 24):
        """
        For files upload. Push one alpha name into redis
        """
        if clear_queue:
            self.clear_queue(self.QUEUE_KEY_PRODUCER_P)
        uuid_list = []
        uuid_info = dict()
        result_info = {1: "accepted", 2: "watched", 3: "rejected", -1: "error", -3: "kill", -4: "duplicated",
                       -2: "timeout"}
        for df in dataframe_list:
            alpha_name = df.columns[0]
            uuid_ = str(uuid.uuid3(uuid.NAMESPACE_DNS, researcher + "_" + alpha_name))
            uuid_info[uuid_] = dict()
            uuid_info[uuid_]['alpha_name'] = alpha_name
            uuid_list.append(uuid_)
            dbItem = {
                "uname": researcher,
                "alpha_name": alpha_name,
                "feature_data": self.context.serialize(df).to_buffer().to_pybytes(),
                "category": category,
                "uuid": uuid_,
                "jobid": job_id,
                "project_name": project_name,
                "branch_name": branch_name,
                "time_out": time_out
            }
            length_queue = self.redis.llen(self.QUEUE_KEY_PRODUCER_P)
            while length_queue >= self.configs.QUEUE_LENGTH:
                time.sleep(10)
                self.logger.info(
                    f'Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} waiting! '
                    f'length of queue is {length_queue}')
                print(
                    f'Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} waiting! '
                    f'length of queue is {length_queue}')
                length_queue = self.redis.llen(self.QUEUE_KEY_PRODUCER_P)

            self.redis.lpush(self.QUEUE_KEY_PRODUCER_P, pickle.dumps(dbItem))
            self.set_uuid(uuid_, 0, time_out=time_out)

        time_start = time.time()
        while time.time() - time_start <= time_out:
            for uuid_ in uuid_list:
                r = self.redis.get(uuid_).decode("utf-8")
                if int(r) in (1, 2, 3,):
                    # 1. ACCEPTED | 2. WATCHED | 3. REJECTED
                    # update later... -> 1. ACCEPTED-UNCONFIRMED | 2. WATCHED-UNCONFIRMED | 3. REJECTED
                    flag = int(r)
                    uuid_list.remove(uuid_)
                    msg = f'Test of {researcher}-{uuid_info[uuid_]["alpha_name"]} ' \
                          f'finished! | result: {result_info[flag]}'
                    uuid_info[uuid_]['result'] = flag
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    self.logger.info(msg)

                elif int(r) == -1:  # ERROR
                    uuid_list.remove(uuid_)
                    msg = f'Test of {researcher}-{uuid_info[uuid_]["alpha_name"]} failed!'
                    uuid_info[uuid_]['result'] = -1
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    self.logger.info(msg)

                elif int(r) == -3:  # KILL
                    uuid_list.remove(uuid_)
                    msg = f'Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} has been killed!'
                    uuid_info[uuid_]['result'] = -3
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    self.logger.info(msg)

                elif int(r) == -4:  # DUPLICATE
                    uuid_list.remove(uuid_)
                    msg = f'Feature {researcher}-{uuid_info[uuid_]["alpha_name"]} already enrolled, no action.'
                    uuid_info[uuid_]['result'] = -4
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    self.logger.info(msg)

                else:
                    continue

            if len(uuid_list) == 0:
                self.logger.info(
                    f'Operation of {researcher}-{file_path} finished!')
                break
            else:
                time.sleep(5)

        else:
            # clear queue, time_out
            for uuid_ in uuid_list:
                self.redis.delete(uuid_)
                msg = f'Time out for operation of {researcher}-{uuid_info[uuid_]["alpha_name"]}'
                uuid_info[uuid_]['result'] = -2
                uuid_info[uuid_]['message'] = msg
                self.logger.info(msg)
            self.logger.info(
                f'Time out for operation of {researcher}-{file_path}. time out list: '
                f'{",".join([uuid_info[item]["alpha_name"] for item in uuid_list])}')
        return uuid_info

    def add_factors_to_redis_for_dataframe_update(self, researcher, dataframe_list, file_path, clear_queue=False):
        """
        For files upload. Push one alpha name into redis
        """
        if clear_queue:
            self.clear_queue(self.QUEUE_KEY_PRODUCER_UPDATE_P)
        uuid_list = []
        uuid_info = dict()
        for df in dataframe_list:
            alpha_name = df.columns[0]
            uuid_ = str(uuid.uuid3(uuid.NAMESPACE_DNS, researcher + "_" + alpha_name))
            uuid_info[uuid_] = dict()
            uuid_info[uuid_]['alpha_name'] = alpha_name
            uuid_list.append(uuid_)
            dbItem = {
                "uname": researcher,
                "alpha_name": alpha_name,
                "feature_data": self.context.serialize(df).to_buffer().to_pybytes(),
                "uuid": uuid_
            }
            self.redis.lpush(self.QUEUE_KEY_PRODUCER_UPDATE_P, pickle.dumps(dbItem))
            self.set_uuid(uuid_, 0)

        time_start = time.time()
        while time.time() - time_start <= 3600 * 24:
            for uuid_ in uuid_list:
                r = self.redis.get(uuid_).decode("utf-8")
                if int(r) == 1:  # FINISHED
                    uuid_list.remove(uuid_)
                    msg = f'Test of {researcher}-{uuid_info[uuid_]["alpha_name"]} finished!'
                    uuid_info[uuid_]['result'] = 1
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    # self.logger.info(
                    #     f'Operation of {researcher}-{uuid_info[uuid_]} finished!')

                elif int(r) == -1:  # ERROR
                    uuid_list.remove(uuid_)
                    msg = f'Test of {researcher}-{uuid_info[uuid_]["alpha_name"]} failed!'
                    uuid_info[uuid_]['result'] = -1
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)
                    # self.logger.info(
                    #     f'Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} failed!')

                elif int(r) == -3:  # KILL
                    uuid_list.remove(uuid_)
                    msg = f'Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} has been killed!'
                    uuid_info[uuid_]['result'] = -3
                    uuid_info[uuid_]['message'] = msg
                    self.redis.delete(uuid_)

                else:
                    continue

            if len(uuid_list) == 0:
                self.logger.info(
                    f'Operation of {researcher}-{file_path} finished!')
                break
            else:
                time.sleep(5)

        else:
            # clear queue, time_out
            for uuid_ in uuid_list:
                uuid_info[uuid_]['result'] = -2  # Timeout
                msg = f'Time Out! Operation of {researcher}-{uuid_info[uuid_]["alpha_name"]} failed!'
                uuid_info[uuid_]['message'] = msg
                self.redis.delete(uuid_)
            # self.logger.info(
            #     f'Time out for operation of {researcher}-{file_path}. time out list: '
            #     f'{",".join([uuid_info[item]["alpha_name"] for item in uuid_list])}')

        return uuid_info

    def get_one_factor_from_redis_for_file(self):
        r = self.redis.brpop(self.QUEUE_KEY_PRODUCER_P, timeout=10)
        if len(r) == 2:
            result = json.loads(r[1].decode("utf-8"))
            return result
        else:
            return None

    def get_one_factor_from_redis_for_dataframe(self):
        r = self.redis.brpop(self.QUEUE_KEY_PRODUCER_P, timeout=10)
        if r is None:
            return None
        else:
            if len(r) == 2:
                # result = json.loads(r[1].decode("utf-8"))
                result = pickle.loads(r[1])
                return result
            else:
                return None

    def get_one_factor_from_redis_for_dataframe_update(self):
        r = self.redis.brpop(self.QUEUE_KEY_PRODUCER_UPDATE_P, timeout=10)
        if r is None:
            return None
        else:
            if len(r) == 2:
                # result = json.loads(r[1].decode("utf-8"))
                result = pickle.loads(r[1])
                return result
            else:
                return None

    def clear_queue(self, queue_name):
        while True:
            r = self.redis.brpop(queue_name, timeout=10)
            if r is None:
                return
            else:
                continue


class FeatureOps(FeatureAnalysis):
    def __init__(self, options, mysql_info=TEST_FACTORS_DAILY_TI0, feature_path=None, logger=None,
                 workers=mp.cpu_count(), check_corr=False, check_table=False):
        # options = get_global_configs()
        super().__init__(options=options, mysql_info=mysql_info,
                         feature_path=feature_path, logger=None, workers=workers)
        self.load_corr_matrix()
        self.db_api = MysqlOps(mysql_info=mysql_info, logger=None)
        if check_table:
            self.db_api.check_table(return_table=True, ic_table=True)
        if check_corr:
            self.check_corr_matrix()

        logger = get_elk_logger("platform_correlation_service_test", console=False)
        self.dist = Distribute(options, logger=logger)
        self.return_path = os.path.join(options.return_path_daily, '1min')
        # if logger is None:
        #     self.logger = get_elk_logger("platform_upload_service")
        # else:
        #     self.logger = logger
        self.logger = logger

    def check_corr_matrix(self, start_time=None, end_time=None):
        if len(self.features_in_db) == 0:
            if os.path.exists(self.options.corr_path):
                os.remove(self.options.corr_path)
                return
            else:
                return

        if self.corr_matrix is None:
            stock_list_in_corr = set()
        else:
            stock_list_in_corr = set(self.corr_matrix.columns.tolist())
        stock_list_in_db = set(self.db_features_info[self.db_features_info['status'] == 'accepted']
                               ['table_name'].values.tolist())
        if not stock_list_in_corr == stock_list_in_db:
            stock_missing = list(stock_list_in_db.difference(stock_list_in_corr))
            if len(stock_missing) != 0:
                if self.logger is not None:
                    self.logger.error(f"Features: {stock_missing} not in corr matrix")
                for item in stock_missing:
                    self.load_feature_from_db(item, start_time, end_time, universe='Investable',
                                              timedelta=None, transformer=None, cover_rate=False)

                    if self.corr_matrix is None:
                        self.corr_matrix = self.feature_data.copy()
                    else:
                        self.corr_matrix = self.corr_matrix.join(self.feature_data, how='outer')

            stock_duplicates = list(stock_list_in_corr.difference(stock_list_in_db))
            if len(stock_duplicates) != 0:
                if self.logger is not None:
                    self.logger.error(f"Features: {stock_duplicates} are in corr matrix but not in feature DB.")

            self.corr_matrix = self.corr_matrix.reindex(columns=stock_list_in_db)
            self.corr_matrix.to_parquet(self.options.corr_path)

    def update_to_watch_list(self, alpha_name, change_reason):
        self.corr_matrix = self.corr_matrix.drop(columns=alpha_name)
        self.corr_matrix = self.corr_matrix.join(
            self.feature_data.loc[(self.feature_data.index.get_level_values(0) <= self.corr_end_time) &
                                  (self.feature_data.index.get_level_values(0) >= self.corr_start_time)], how='outer')
        # write corr_matrix
        self.corr_matrix.to_parquet(self.options.corr_path)
        self.db_api.update_to_watch(alpha_name, change_reason)

    # 8.2 update alpha_return_info
    def update_alpha_return_info(self, alpha_names=None, benchmark='HS300'):
        conn = self.db_api.pool.connection()
        curs = conn.cursor()

        factor_all_table = self.db_features_info
        if alpha_names is None:
            alpha_list = self.features_in_db
        else:
            assert isinstance(alpha_names, list), "alpha_names must be a list."
            alpha_list = alpha_names

        if self.trading_days is not None:
            end_date = self.trading_days[-1]
        else:
            # end_date = datetime.date.today()
            end_date = '2021-07-01'
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
        start_date = datetime.date(end_date.year - 3, end_date.month, end_date.day)

        configs = self.options
        configs.data_format = "dataframe"
        configs.daily_data_missing_allowed = True
        configs.trade_period = 0
        configs.stock_percentage = True
        if benchmark in ('HS300', 'ZZ500',):
            configs.stock_num = 0.2
            configs.benchmark = benchmark
        else:
            configs.benchmark = 'ZZ500'
            configs.stock_num = 0.1
        configs.transmission_rate = 0.0
        configs.tax_rate = 0.000
        configs.bt_price = 'close'
        configs.universe = "All"

        for alpha_name in alpha_list:
            self.load_feature_data_from_file(alpha_name, start_date, get_next_trading_day(end_date),
                                             timedelta=self.time_delta, universe=benchmark, normalized=False)
            df = self.feature_data.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d').astype(int)
            trade_direct = factor_all_table.loc[factor_all_table['table_name'] == alpha_name, 'trade_direction'].iloc[0]

            if trade_direct == -1:
                df[alpha_name] = df[alpha_name] * trade_direct

            df.set_index('timestamp', inplace=True)

            configs.ti = self.ti

            return_result_dict = dict()
            for trade_type in ('long-only', 'short-only'):
                configs.trading_type = trade_type
                try:
                    bt = BTDaily(configs, start_date=start_date, end_date=end_date, logger=self.logger)
                    bt.feed_data(df)
                    bt.run()
                    return_daily, _ = bt.evaluate(evalRange=None, verbose=False)

                    for length in (21, 63, 126, 252, 753):
                        result_temp = return_daily.tail(length)
                        av = result_temp['alpha_value']
                        _sharpe = calc_sharpe_ratio(av.tolist())
                        _return = (av.iloc[-1] - av.iloc[0]) / av.iloc[0]
                        _return_yearly = np.power(np.power(1 + _return, 1 / length), 252) - 1
                        _turnover = result_temp['turnover'].mean()
                        return_result_dict[trade_type + '_' + str(length) + '_ret'] = _return_yearly * 100
                        return_result_dict[trade_type + '_' + str(length) + '_turnover'] = _turnover
                        return_result_dict[trade_type + '_' + str(length) + '_sharpe'] = _sharpe
                        return_result_dict[trade_type + '_' + str(length) + '_data'] = av
                except AssertionError as err:
                    if "No valid stock when calculating portfolio return" in str(err):
                        for length in (21, 63, 126, 252, 753):
                            return_result_dict[trade_type + '_' + str(length) + '_ret'] = np.nan
                            return_result_dict[trade_type + '_' + str(length) + '_turnover'] = np.nan
                            return_result_dict[trade_type + '_' + str(length) + '_sharpe'] = np.nan

            for length in (21, 63, 126, 252, 753):
                return_result_dict['long-short' + '_' + str(length) + '_ret'] = \
                    return_result_dict['long-only' + '_' + str(length) + '_ret'] + \
                    return_result_dict['short-only' + '_' + str(length) + '_ret']
                return_result_dict['long-short' + '_' + str(length) + '_turnover'] = \
                    return_result_dict['long-only' + '_' + str(length) + '_turnover'] + \
                    return_result_dict['short-only' + '_' + str(length) + '_turnover']

                if np.isnan(return_result_dict['long-only' + '_' + str(length) + '_ret']) or \
                        np.isnan(return_result_dict['short-only' + '_' + str(length) + '_ret']):
                    return_result_dict['long-short' + '_' + str(length) + '_sharpe'] = np.nan
                else:
                    av_all = \
                        return_result_dict['long-only' + '_' + str(length) + '_data'] + \
                        return_result_dict['short-only' + '_' + str(length) + '_data']
                    return_result_dict['long-short' + '_' + str(length) + '_sharpe'] = \
                        calc_sharpe_ratio(av_all.tolist())

            return_result_list = []
            item_target = ['long-only_21_ret', 'short-only_21_ret', 'long-short_21_ret', 'long-only_21_turnover',
                           'short-only_21_turnover', 'long-short_21_turnover', 'long-only_21_sharpe',
                           'short-only_21_sharpe', 'long-short_21_sharpe',
                           'long-only_63_ret', 'short-only_63_ret', 'long-short_63_ret', 'long-only_63_turnover',
                           'short-only_63_turnover', 'long-short_63_turnover', 'long-only_63_sharpe',
                           'short-only_63_sharpe', 'long-short_63_sharpe',
                           'long-only_126_ret', 'short-only_126_ret', 'long-short_126_ret', 'long-only_126_turnover',
                           'short-only_126_turnover', 'long-short_126_turnover', 'long-only_126_sharpe',
                           'short-only_126_sharpe', 'long-short_126_sharpe',
                           'long-only_252_ret', 'short-only_252_ret', 'long-short_252_ret', 'long-only_252_turnover',
                           'short-only_252_turnover', 'long-short_252_turnover', 'long-only_252_sharpe',
                           'short-only_252_sharpe', 'long-short_252_sharpe',
                           'long-only_753_ret', 'short-only_753_ret', 'long-short_753_ret', 'long-only_753_turnover',
                           'short-only_753_turnover', 'long-short_753_turnover', 'long-only_753_sharpe',
                           'short-only_753_sharpe', 'long-short_753_sharpe']
            for item in item_target:
                if np.isnan(return_result_dict[item]):
                    return_result_list.append(None)
                else:
                    return_result_list.append(str(round(return_result_dict[item], 5)))
            return_result_list.insert(0, alpha_name)

            sql = f"""
            replace into 
                alpha_return_info_{benchmark.lower()} (alpha_name, rec_1m_long_ret, rec_1m_short_ret, rec_1m_ls_ret, 
                rec_1m_long_turnover, rec_1m_short_turnover, rec_1m_ls_turnover, rec_1m_long_sharpe, 
                rec_1m_short_sharpe, rec_1m_ls_sharpe, rec_3m_long_ret, rec_3m_short_ret, rec_3m_ls_ret, 
                rec_3m_long_turnover, rec_3m_short_turnover, rec_3m_ls_turnover, rec_3m_long_sharpe, 
                rec_3m_short_sharpe, rec_3m_ls_sharpe, rec_6m_long_ret, rec_6m_short_ret, rec_6m_ls_ret, 
                rec_6m_long_turnover, rec_6m_short_turnover, rec_6m_ls_turnover, rec_6m_long_sharpe, 
                rec_6m_short_sharpe, rec_6m_ls_sharpe, rec_1y_long_ret, rec_1y_short_ret, rec_1y_ls_ret, 
                rec_1y_long_turnover, rec_1y_short_turnover, rec_1y_ls_turnover, rec_1y_long_sharpe, 
                rec_1y_short_sharpe, rec_1y_ls_sharpe, rec_3y_long_ret, rec_3y_short_ret, rec_3y_ls_ret, 
                rec_3y_long_turnover, rec_3y_short_turnover, rec_3y_ls_turnover, rec_3y_long_sharpe, 
                rec_3y_short_sharpe, rec_3y_ls_sharpe) 
            values 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s);"""

            try:
                curs.execute(sql, tuple(return_result_list))
            except Exception as e:
                # print(f"""{datetime.datetime.now()}, insert into {table_name} error, error is {e}.""")
                if self.logger is not None:
                    self.db_api.logger.error(f"Test of {alpha_name} fails, error message: {e}")
                conn.rollback()
                self.reset()
            else:
                conn.commit()

            self.reset()

        conn.close()

    # 8.2 update alpha_return_info
    def update_return_after_cost(self, alpha_names=None, benchmark='HS300'):
        conn = self.db_api.pool.connection()
        assert isinstance(alpha_names, list), "alpha_names must be a list."

        columns = ['alpha_name']
        for i in ('1m', '3m', '6m', '1y', '3y'):
            for j in ('long', 'short', 'ls'):
                for k in ('ret', 'turnover'):
                    columns.append(f"rec_{i}_{j}_{k}")

        columns_str = ','.join(columns)
        if alpha_names is None:
            sql = f"""select {columns_str} from alpha_return_info_{benchmark.lower()};"""
        else:
            string = "','".join(alpha_names)
            sql = f"""select {columns_str} from alpha_return_info_{benchmark.lower()} WHERE alpha_name in """ + \
                  f"('{string}');"
        df = pd.read_sql(sql, conn)

        target_columns = ['alpha_name']
        for i in ('1m', '3m', '6m', '1y', '3y'):
            for j in ('long', 'short', 'ls'):
                df[f"rec_{i}_{j}_ret_afc_1"] = \
                    df[f"rec_{i}_{j}_ret"] - (0.05 + 0.1) * df[f"rec_{i}_{j}_turnover"] * 252
                df[f"rec_{i}_{j}_ret_afc_2"] = \
                    df[f"rec_{i}_{j}_ret"] - (0.15 + 0.1) * df[f"rec_{i}_{j}_turnover"] * 252
                target_columns.append(f"rec_{i}_{j}_ret_afc_1")
                target_columns.append(f"rec_{i}_{j}_ret_afc_2")

        df_target = df.reindex(columns=target_columns)
        try:
            self.db_api.insert_to_afc(df_target, benchmark)
        except Exception as e:
            # print(f"""{datetime.datetime.now()}, insert into {table_name} error, error is {e}.""")
            if self.logger is not None:
                self.db_api.logger.error(f"Update {benchmark} cost table fails, error message: {e}")
            conn.rollback()
            self.reset()
        else:
            conn.commit()

        conn.close()

    def get_correlation_dist(self):
        assert self.feature_data is not None, "feature data must be set at first!"
        self.logger.info(
            f'Feature of {self.feature_name} has been sent.')
        md5, corr_matrix = self.dist.to_redis_upload_dataframe(self.feature_data)
        return md5, corr_matrix

    def get_correlations_with_factor_lib_dist(self, feature_name, verbose=True, output=None):
        """
        :param
        method: {'pearson', 'kendall', 'spearman'} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays
        """
        md5, corr_table_result = self.get_correlation_dist()

        flag, msg, names = is_correlation_accepted(feature_name, corr_table_result)

        if flag == 0:
            if self.logger is not None:
                self.logger.warning(msg)
            if verbose:
                print(msg)
            return False, names, md5, msg

        elif flag == 1:
            if self.logger is not None:
                self.logger.info(msg)
            if verbose:
                print(msg)
            return True, names, md5, msg

        if flag == 4:
            if self.logger is not None:
                self.logger.error(msg)
            if verbose:
                print(msg)
            return False, names, md5, msg

        else:
            if output is not None:
                output_file = os.path.join(output, feature_name, 'corr_table_{}.csv'.format(feature_name))
                corr_table_result.to_csv(output_file, sep=',', mode='w', header=True, index=True, encoding='utf-8')
            if verbose:
                corr_table_result.sort_values(by='correlation', inplace=True)
                print('\nCorrelations for feature "{}":'.format(feature_name))
                print(tabulate(corr_table_result, headers=['alpha_name'] + corr_table_result.columns.tolist(),
                               tablefmt='grid', floatfmt=".3f", stralign="center", numalign="center"))

            if flag == 2:
                if self.logger is not None:
                    self.logger.info(msg)
                if verbose:
                    print(msg)
                return True, names, md5, msg

            else:
                if self.logger is not None:
                    self.logger.info(msg)
                if verbose:
                    print(msg)
                return False, names, md5, msg

    def handle_recent_ic_ir_info(self, alpha_name, ic_table, benchmark):
        result_list = []
        for period in self.options.EVAL_IC_PERIODS:
            target_column = [item for item in ic_table.columns if item.endswith(period)]
            ic_ir_recent_table = self.ic_table.reindex(columns=target_column)
            ic_result_list = []
            for length in self.options.EVAL_RECENT_PERIODS_IN_DAYS:
                length = int(length)
                if len(ic_ir_recent_table) >= length:
                    result_temp = ic_ir_recent_table.tail(length)
                    ic_result = result_temp.mean()
                    ir_result = result_temp.mean() / result_temp.std()
                    ic_result_list.extend([ic_result.iloc[0].round(5), ir_result.iloc[0].round(5)])
                else:
                    ic_result_list.extend([np.nan, np.nan])
            ic_result_list.insert(0, alpha_name)
            ic_result_list.insert(1, benchmark)
            if period.endswith('m'):
                ic_result_list.insert(2, period + 'in')
            else:
                ic_result_list.insert(2, period)
            result_list.append(ic_result_list)
        df = pd.DataFrame(result_list,
                          columns=['alpha_name', 'universe', 'period_in_day', 'rec_1m_IC', 'rec_1m_IR',
                                   'rec_3m_IC', 'rec_3m_IR', 'rec_6m_IC', 'rec_6m_IR', 'rec_1y_IC', 'rec_1y_IR',
                                   'rec_2y_IC', 'rec_2y_IR', 'rec_3y_IC', 'rec_3y_IR']).dropna()
        self.db_api.insert_to_alpha_related_table(table_names=['alpha_recent_icir'], dataframe_list=[df])

    def handle_daily_ic_info(self, alpha_name, ic_table, benchmark):
        table_names = []
        dataframe_list = []
        rename_dict = {'timestamp': 'trading_time'}
        for item in ic_table.columns:
            if item.endswith('m'):
                item_target = item + 'in'
            else:
                item_target = item
            rename_dict[item] = item_target.replace(alpha_name + '_', "")

        for bt_price in self.options.EVAL_IC_PRICES:
            if bt_price in ('twap', 'vwap'):
                for tp in self.options.EVAL_TRADING_PERIODS:
                    table_names.append(f"alpha_daily_ic_{benchmark}_{bt_price}_{tp}")
                    df_result = ic_table.reset_index().rename(columns=rename_dict)
                    df_result.insert(1, 'alpha_name', alpha_name)
                    dataframe_list.append(df_result.dropna())
            else:
                table_names.append(f"alpha_daily_ic_{benchmark}_{bt_price}_0")
                df_result = ic_table.reset_index().rename(columns=rename_dict)
                df_result.insert(1, 'alpha_name', alpha_name)
                dataframe_list.append(df_result.dropna())

        self.db_api.insert_to_alpha_related_table(table_names=table_names, dataframe_list=dataframe_list)

    def _handle_daily_ic_info(self, alpha_name, ic_table, benchmark):
        table_names = []
        dataframe_list = []
        rename_dict = {'timestamp': 'trading_time'}
        for item in ic_table.columns:
            if item.endswith('m'):
                item_target = item + 'in'
            else:
                item_target = item
            rename_dict[item] = item_target.replace(alpha_name + '_', "")

        for bt_price in self.options.EVAL_IC_PRICES:
            if bt_price in ('twap', 'vwap'):
                table_names.append(f"alpha_ic_daily_investable_vwap")
                df_result = ic_table.reset_index().rename(columns=rename_dict)
                df_result.insert(1, 'alpha_name', alpha_name)
                dataframe_list.append(df_result.dropna())
            else:
                table_names.append(f"alpha_daily_ic_{benchmark}_{bt_price}_0")
                df_result = ic_table.reset_index().rename(columns=rename_dict)
                df_result.insert(1, 'alpha_name', alpha_name)
                dataframe_list.append(df_result.dropna())

        self.db_api.insert_to_alpha_related_table(table_names=table_names, dataframe_list=dataframe_list)

    def handle_daily_return_grouped_info(self, alpha_name, negative, benchmark):
        table_names = []
        dataframe_list = []
        grouped = self.feature_data.groupby(level=0)
        df_result_5_rec = None
        df_result_10_rec = None

        # alpha_daily_return_{benchmark}_{bt_price}_{tp}_group5
        for bt_price in self.options.EVAL_RETURN_PRICES:
            if bt_price in ('twap', 'vwap'):
                for tp in self.options.EVAL_TRADING_PERIODS:
                    tp = int(tp)
                    df_result_5, df_holdings = self.get_group_returns(feature_name=alpha_name, negative=negative,
                                                                      group_num=5, bt_price=bt_price, ti=self.ti,
                                                                      trade_period=tp)
                    df_result_5 = df_result_5.reset_index().rename(columns={'timestamp': 'trading_time'})
                    df_result_5['trading_time'] = pd.to_datetime(df_result_5['trading_time'], format='%Y%m%d') + \
                                                  pd.Timedelta(hours=self.time_list[0] // 100,
                                                               minutes=self.time_list[0] % 100)
                    df_result_5.insert(0, "alpha_name", alpha_name)
                    table_names.append(f"alpha_daily_return_{benchmark}_{bt_price}_{tp}_group5")
                    dataframe_list.append(df_result_5)
            else:
                df_result_5, df_holdings = self.get_group_returns(feature_name=alpha_name, negative=negative,
                                                                  group_num=5, bt_price=bt_price, ti=self.ti,
                                                                  trade_period=0)
                df_result_5 = df_result_5.reset_index().rename(columns={'timestamp': 'trading_time'})
                df_result_5['trading_time'] = pd.to_datetime(df_result_5['trading_time'], format='%Y%m%d') + \
                                              pd.Timedelta(hours=self.time_list[0] // 100,
                                                           minutes=self.time_list[0] % 100)
                df_result_5.insert(0, "alpha_name", alpha_name)
                table_names.append(f"alpha_daily_return_{benchmark}_{bt_price}_0_group5")
                dataframe_list.append(df_result_5)

                if bt_price == 'close':
                    df_result_5_rec = df_result_5

        # alpha_daily_return_{benchmark}_holdings_group5
        temp_list = []
        for key in df_holdings[0].keys():
            temp_1 = df_holdings[0][key]
            temp_2 = df_holdings[4][key]
            timestamp_target = pd.Timestamp(str(key)) + pd.Timedelta(hours=self.time_list[0] // 100,
                                                                     minutes=self.time_list[0] % 100)
            daily_df = grouped.get_group(timestamp_target) * -int(negative)
            df_1 = daily_df.loc[daily_df.index.get_level_values(1).isin(temp_1.index)].sort_values(by=alpha_name).head(
                100)
            df_1.insert(1, "direction", 'short')
            df_2 = daily_df.loc[daily_df.index.get_level_values(1).isin(temp_2.index)].sort_values(by=alpha_name,
                                                                                                   ascending=False).head(
                100)
            df_2.insert(1, "direction", 'long')
            df_result = pd.concat([df_2, df_1])
            df_result = df_result.reset_index().drop(columns='timestamp')
            df_result.insert(0, 'date', int(key))
            df_result.insert(0, 'alpha_name', alpha_name)
            df_result = df_result.rename(columns={alpha_name: 'alpha_value', 'ticker': 'symbol'})
            temp_list.append(df_result)

        df_holding_result_5 = pd.concat(temp_list)
        table_names.append(f"alpha_daily_return_{benchmark}_holdings_group5")
        dataframe_list.append(df_holding_result_5)

        # alpha_daily_return_{benchmark}_{bt_price}_{tp}_group10
        for bt_price in self.options.EVAL_RETURN_PRICES:
            if bt_price in ('twap', 'vwap'):
                for tp in self.options.EVAL_TRADING_PERIODS:
                    tp = int(tp)
                    df_result_10, df_holdings = self.get_group_returns(feature_name=alpha_name, negative=negative,
                                                                       group_num=10, bt_price=bt_price, ti=self.ti,
                                                                       trade_period=tp)
                    df_result_10 = df_result_10.reset_index().rename(columns={'timestamp': 'trading_time'})
                    df_result_10['trading_time'] = pd.to_datetime(df_result_10['trading_time'], format='%Y%m%d') + \
                                                   pd.Timedelta(hours=self.time_list[0] // 100,
                                                                minutes=self.time_list[0] % 100)
                    df_result_10.insert(0, "alpha_name", alpha_name)
                    table_names.append(f"alpha_daily_return_{benchmark}_{bt_price}_{tp}_group10")
                    dataframe_list.append(df_result_10)
            else:
                df_result_10, df_holdings = self.get_group_returns(feature_name=alpha_name, negative=negative,
                                                                   group_num=10, bt_price=bt_price, ti=self.ti,
                                                                   trade_period=0)
                df_result_10 = df_result_10.reset_index().rename(columns={'timestamp': 'trading_time'})
                df_result_10['trading_time'] = pd.to_datetime(df_result_10['trading_time'], format='%Y%m%d') + \
                                               pd.Timedelta(hours=self.time_list[0] // 100,
                                                            minutes=self.time_list[0] % 100)
                df_result_10.insert(0, "alpha_name", alpha_name)
                table_names.append(f"alpha_daily_return_{benchmark}_{bt_price}_0_group10")
                dataframe_list.append(df_result_10)
                if bt_price == 'close':
                    df_result_10_rec = df_result_10

        # alpha_daily_return_{benchmark}_holdings_group5
        temp_list = []
        for key in df_holdings[0].keys():
            temp_1 = df_holdings[0][key]
            temp_2 = df_holdings[9][key]
            timestamp_target = pd.Timestamp(str(key)) + pd.Timedelta(hours=self.time_list[0] // 100,
                                                                     minutes=self.time_list[0] % 100)
            daily_df = grouped.get_group(timestamp_target) * -int(negative)
            df_1 = daily_df.loc[daily_df.index.get_level_values(1).isin(temp_1.index)].sort_values(by=alpha_name).head(
                100)
            df_1.insert(1, "direction", 'short')
            df_2 = daily_df.loc[daily_df.index.get_level_values(1).isin(temp_2.index)].sort_values(by=alpha_name,
                                                                                                   ascending=False).head(
                100)
            df_2.insert(1, "direction", 'long')
            df_result = pd.concat([df_2, df_1])
            df_result = df_result.reset_index().drop(columns='timestamp')
            df_result.insert(0, 'date', int(key))
            df_result.insert(0, 'alpha_name', alpha_name)
            df_result = df_result.rename(columns={alpha_name: 'alpha_value', 'ticker': 'symbol'})
            temp_list.append(df_result.dropna())

        df_holding_result_10 = pd.concat(temp_list)
        table_names.append(f"alpha_daily_return_{benchmark}_holdings_group10")
        dataframe_list.append(df_holding_result_10.dropna())

        self.db_api.insert_to_alpha_related_table(table_names=table_names, dataframe_list=dataframe_list)
        return df_result_5_rec, df_result_10_rec

    def handle_recent_returns_info(self, alpha_name, return_table, group_num, benchmark):
        return_result_dict = dict()
        for trade_type in ('long-only', 'short-only'):
            try:
                for length in self.options.EVAL_RECENT_PERIODS_IN_DAYS:
                    length = int(length)
                    if len(return_table) >= length:
                        result_temp = return_table.tail(length)
                        if trade_type == 'long-only':
                            column_name_alpha = f'alpha_group{group_num - 1}'
                            column_name_turnover = f'turnover_group{group_num - 1}'
                        else:
                            column_name_alpha = f'alpha_group0'
                            column_name_turnover = f'turnover_group0'
                        av = result_temp[column_name_alpha]
                        _sharpe = calc_sharpe_ratio(av.tolist())
                        _return = (av.iloc[-1] - av.iloc[0]) / av.iloc[0]
                        _return_yearly = np.power(np.power(1 + _return, 1 / length), 252) - 1
                        _turnover = result_temp[column_name_turnover].mean()
                        return_result_dict[trade_type + '_' + str(length) + '_ret'] = _return_yearly * 100
                        return_result_dict[trade_type + '_' + str(length) + '_turnover'] = _turnover
                        return_result_dict[trade_type + '_' + str(length) + '_sharpe'] = _sharpe
                        return_result_dict[trade_type + '_' + str(length) + '_data'] = av
                    else:
                        return_result_dict[trade_type + '_' + str(length) + '_ret'] = np.nan
                        return_result_dict[trade_type + '_' + str(length) + '_turnover'] = np.nan
                        return_result_dict[trade_type + '_' + str(length) + '_sharpe'] = np.nan

            except AssertionError as err:
                if "No valid stock when calculating portfolio return" in str(err):
                    for length in self.options.EVAL_RECENT_PERIODS_IN_DAYS:
                        return_result_dict[trade_type + '_' + str(length) + '_ret'] = np.nan
                        return_result_dict[trade_type + '_' + str(length) + '_turnover'] = np.nan
                        return_result_dict[trade_type + '_' + str(length) + '_sharpe'] = np.nan

        for length in self.options.EVAL_RECENT_PERIODS_IN_DAYS:
            return_result_dict['long-short' + '_' + str(length) + '_ret'] = \
                return_result_dict['long-only' + '_' + str(length) + '_ret'] + \
                return_result_dict['short-only' + '_' + str(length) + '_ret']
            return_result_dict['long-short' + '_' + str(length) + '_turnover'] = \
                return_result_dict['long-only' + '_' + str(length) + '_turnover'] + \
                return_result_dict['short-only' + '_' + str(length) + '_turnover']

            if np.isnan(return_result_dict['long-only' + '_' + str(length) + '_ret']) or \
                    np.isnan(return_result_dict['short-only' + '_' + str(length) + '_ret']):
                return_result_dict['long-short' + '_' + str(length) + '_sharpe'] = np.nan
            else:
                av_all = \
                    return_result_dict['long-only' + '_' + str(length) + '_data'] + \
                    return_result_dict['short-only' + '_' + str(length) + '_data']
                return_result_dict['long-short' + '_' + str(length) + '_sharpe'] = \
                    calc_sharpe_ratio(av_all.tolist())

        return_result_list = []
        item_target = []
        for day in self.options.EVAL_RECENT_PERIODS_IN_DAYS:
            item_target.extend([f'long-only_{day}_ret', f'short-only_{day}_ret', f'long-short_{day}_ret',
                                f'long-only_{day}_turnover', f'short-only_{day}_turnover',
                                f'long-short_{day}_turnover', f'long-only_{day}_sharpe',
                                f'short-only_{day}_sharpe', f'long-short_{day}_sharpe'])

        for item in item_target:
            if np.isnan(return_result_dict[item]):
                return_result_list.append(np.nan)
            else:
                return_result_list.append(round(return_result_dict[item], 5))
        return_result_list.insert(0, alpha_name)
        return_result_list.insert(1, benchmark)

        columns_target = ['alpha_name', 'universe',
                          'rec_1m_long_ret', 'rec_1m_short_ret', 'rec_1m_ls_ret', 'rec_1m_long_turnover',
                          'rec_1m_short_turnover', 'rec_1m_ls_turnover', 'rec_1m_long_sharpe',
                          'rec_1m_short_sharpe', 'rec_1m_ls_sharpe',
                          'rec_3m_long_ret', 'rec_3m_short_ret', 'rec_3m_ls_ret', 'rec_3m_long_turnover',
                          'rec_3m_short_turnover', 'rec_3m_ls_turnover', 'rec_3m_long_sharpe',
                          'rec_3m_short_sharpe', 'rec_3m_ls_sharpe',
                          'rec_6m_long_ret', 'rec_6m_short_ret', 'rec_6m_ls_ret', 'rec_6m_long_turnover',
                          'rec_6m_short_turnover', 'rec_6m_ls_turnover', 'rec_6m_long_sharpe',
                          'rec_6m_short_sharpe', 'rec_6m_ls_sharpe',
                          'rec_1y_long_ret', 'rec_1y_short_ret', 'rec_1y_ls_ret', 'rec_1y_long_turnover',
                          'rec_1y_short_turnover', 'rec_1y_ls_turnover', 'rec_1y_long_sharpe',
                          'rec_1y_short_sharpe', 'rec_1y_ls_sharpe',
                          'rec_2y_long_ret', 'rec_2y_short_ret', 'rec_2y_ls_ret', 'rec_2y_long_turnover',
                          'rec_2y_short_turnover', 'rec_2y_ls_turnover', 'rec_2y_long_sharpe',
                          'rec_2y_short_sharpe', 'rec_2y_ls_sharpe',
                          'rec_3y_long_ret', 'rec_3y_short_ret', 'rec_3y_ls_ret', 'rec_3y_long_turnover',
                          'rec_3y_short_turnover', 'rec_3y_ls_turnover', 'rec_3y_long_sharpe',
                          'rec_3y_short_sharpe', 'rec_3y_ls_sharpe']
        result_df = pd.DataFrame([return_result_list], columns=columns_target).dropna()

        self.db_api.insert_to_alpha_related_table(table_names=["alpha_recent_returns"], dataframe_list=[result_df])
        return result_df

    def handle_recent_return_after_cost(self, return_table):
        target_columns = ['alpha_name', 'universe']
        for i in ('1m', '3m', '6m', '1y', '2y', '3y'):
            for j in ('long', 'short', 'ls'):
                return_table[f"rec_{i}_{j}_ret_afc_1"] = \
                    return_table[f"rec_{i}_{j}_ret"] - (0.05 + 0.1) * return_table[f"rec_{i}_{j}_turnover"] * 252
                return_table[f"rec_{i}_{j}_ret_afc_2"] = \
                    return_table[f"rec_{i}_{j}_ret"] - (0.15 + 0.1) * return_table[f"rec_{i}_{j}_turnover"] * 252
                target_columns.append(f"rec_{i}_{j}_ret_afc_1")
                target_columns.append(f"rec_{i}_{j}_ret_afc_2")

        result_df = return_table.reindex(columns=target_columns).dropna()
        self.db_api.insert_to_alpha_related_table(table_names=["alpha_recent_return_after_cost"],
                                                  dataframe_list=[result_df])

    def update_related_info_from_file_dist(self, alpha_name, trading_direction, benchmark='Investable'):
        # update alpha icir info
        if self.trading_days is not None:
            end_date = self.trading_days[-1]
        else:
            # end_date = datetime.date.today()
            end_date = '2021-07-01'

        self.load_feature_from_file(alpha_name, self.options.FACTOR_START_DATE, end_date, timedelta=None,
                                    universe=benchmark, transformer=norm.standard_scale, cover_rate=False)
        self.load_return_data()
        self.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman',
                                 period=(
                                     '1m', '5m', '15m', '30m', '60m', '120m', '1d', '2d', '3d', '4d', '5d', '10d',
                                     '20d'))

        # ======= update ic ir info ==============
        # update alpha_recent_icir
        self.handle_recent_ic_ir_info(alpha_name, self.ic_table, benchmark)

        # update alpha_daily_ic_{benchmark}_{bt_price}_{tp}
        self.handle_daily_ic_info(alpha_name, self.ic_table, benchmark)

        #  ======= update return info ==============
        if trading_direction == -1:
            negative = True
        else:
            negative = False
        # update alpha_daily_return &  update alpha_daily_holdings
        df_result_5, df_result_10 = self.handle_daily_return_grouped_info(alpha_name, negative, benchmark)

        # update alpha_recent_returns
        if benchmark in ('HS300', 'ZZ500',):
            return_table = df_result_5
            group_num = 5
        else:
            return_table = df_result_10
            group_num = 10
        recent_return_table = self.handle_recent_returns_info(alpha_name, return_table, group_num, benchmark)

        # update_return_after_cost_info
        self.handle_recent_return_after_cost(recent_return_table)

    def update_related_info_from_dataframe_dist(self, alpha_name, df, trading_direction, benchmark='Investable'):
        # update alpha icir info
        if self.trading_days is not None:
            end_date = self.trading_days[-1]
        else:
            # end_date = datetime.date.today()
            end_date = '2021-07-01'

        self.load_feature_from_dataframe(df, timedelta=None,
                                         universe=benchmark, transformer=norm.rank, cover_rate=False)
        self.load_return_data()
        self.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman',
                                 period=(
                                     '1m', '5m', '15m', '30m', '60m', '120m', '1d', '2d', '3d', '4d', '5d', '10d',
                                     '20d'))

        # ======= update ic ir info ==============
        # update alpha_recent_icir
        self.handle_recent_ic_ir_info(alpha_name, self.ic_table, benchmark)

        # update alpha_daily_ic_{benchmark}_{bt_price}_{tp}
        self.handle_daily_ic_info(alpha_name, self.ic_table, benchmark)

        #  ======= update return info ==============
        if trading_direction == -1:
            negative = True
        else:
            negative = False
        # update alpha_daily_return &  update alpha_daily_holdings
        df_result_5, df_result_10 = self.handle_daily_return_grouped_info(alpha_name, negative, benchmark)

        # update alpha_recent_returns
        if benchmark in ('HS300', 'ZZ500',):
            return_table = df_result_5
            group_num = 5
        else:
            return_table = df_result_10
            group_num = 10
        recent_return_table = self.handle_recent_returns_info(alpha_name, return_table, group_num, benchmark)

        # update_return_after_cost_info
        self.handle_recent_return_after_cost(recent_return_table)

    def distribute_upload_feature_from_file(self, researcher, alpha_name, start_date, end_date, universe='Investable',
                                            timedelta=None, transformer=norm.standard_scale, output=None,
                                            critical_value=None, verbose=False):
        assert self.pathway is not None, \
            "Assign initial parameter `feature_path` first!"

        # parameters
        status = 'rejected'
        change_reason = None
        send_flag = False
        status_json = dict()
        status_json["status"] = 'no_action'
        _periods = self.options.EVAL_PERIODS

        # load eval data and check cover rate
        _, cover_ratio_dict = self.load_feature_from_file(alpha_name, start_date, end_date, universe=universe,
                                                          timedelta=timedelta, transformer=transformer, cover_rate=True)
        cover_ratio = cover_ratio_dict[alpha_name]
        if cover_ratio < self.options.COVER_RATE_LIMIT:
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}, cover_ratio is too low.")
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason="cover_ratio is too low.")
            return status

        # eval IC
        self.load_return_data(price='close')
        ic_flag, trading_direction = self.test_ic(alpha_name, output=output, verbose=verbose)
        if trading_direction is None:
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}, cannot calculate trading direction.")
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio,
                                          change_reason="cannot calculate trading direction.")
            return status

        if ic_flag:
            corr_flag, names, md5, _ = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name,
                                                                                  verbose=verbose)
            # enter factor_db
            if corr_flag:
                status = 'accepted'
                status_json["status"] = 'accepted_no_replace'
            else:
                if names is None:
                    status_json = json.dumps(status_json)
                    self.dist.to_redis_feature_status(md5, status_json)
                    self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)
                    return status
                else:
                    # compare ic value, correlation eval fail
                    _, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
                    alphas_in_db = dict()
                    for name in names:
                        ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                        return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                        alphas_in_db[name] = dict()
                        alphas_in_db[name]['ic'] = ic_target
                        alphas_in_db[name]['ret'] = return_target
                    status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db,
                                                                                    return_info,
                                                                                    self.ic_table)
                    if status == 'accepted' and replace_reason:
                        self.db_api.update_to_watch(names, replace_reason)
                        status_json["status"] = 'accepted_with_replace'
                        status_json["alpha_name"] = names
                    else:
                        pass

            # set redis flag
            send_flag = True

        else:
            return_flag, return_info = self.test_return(alpha_name, trading_direction, verbose=False, output=output)

            if return_flag:
                corr_flag, names, md5, _ = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name,
                                                                                      verbose=False)

                if corr_flag:
                    status = 'accepted'
                    status_json["status"] = 'accepted_no_replace'
                else:
                    if names is None:
                        status_json = json.dumps(status_json)
                        self.dist.to_redis_feature_status(md5, status_json)
                        self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)
                        return status
                    else:
                        alphas_in_db = dict()
                        for name in names:
                            ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                            return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                            alphas_in_db[name] = dict()
                            alphas_in_db[name]['ic'] = ic_target
                            alphas_in_db[name]['ret'] = return_target
                        status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db,
                                                                                        return_info,
                                                                                        self.ic_table)
                    if status == 'accepted' and replace_reason:
                        self.db_api.update_to_watch(names, replace_reason)
                    status_json["status"] = 'accepted_with_replace'
                    status_json["alpha_name"] = names

                # set redis flag
                send_flag = True

        # update table list
        if status in ('accepted', 'watched',):
            self.db_api.register(researcher, alpha_name, trading_direction, cover_ratio, critical_value,
                                 status=status, change_reason=change_reason)
            self.update_related_info_from_file_dist(alpha_name, trading_direction, benchmark='Investable')
        else:
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)

        # send redis here
        if send_flag:
            status_json = json.dumps(status_json)
            self.dist.to_redis_feature_status(md5, status_json)
            self.logger.info(
                f'Operation for corr_matrix of {self.feature_name} has been send.')

        # insert raw data into db
        if status in ('accepted', 'watched',):
            self.load_feature_from_file(alpha_name, self.options.FACTOR_START_DATE, end_date, transformer=None,
                                        timedelta=timedelta, universe='All')
            df_format = self.feature_data.reset_index()
            df_format['timestamp'] = df_format['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_format.rename(columns={'timestamp': "trading_time",
                                      'ticker': 'symbol', alpha_name: "alpha_value"}, inplace=True)
            self.db_api.insert_to_db(df_format.dropna(subset=["alpha_value"]), alpha_name)
            benchmarks = self.options.EVAL_UNIVERSES.copy()
            benchmarks.remove('Investable')
            for bench in benchmarks:
                self.update_related_info_from_file_dist(alpha_name, trading_direction, benchmark=bench)
        if self.logger is not None:
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}")
        return status

    def distribute_upload_feature_from_dataframe(self, researcher, feature_data,
                                                 universe='Investable', category='PV',
                                                 timedelta=None, output=None,
                                                 critical_value=None, verbose=False):

        # parameters
        status = 'rejected'
        change_reason = None
        send_flag = False
        status_json = dict()
        status_json["status"] = 'no_action'
        _periods = self.options.EVAL_PERIODS

        # load eval data and check cover rate
        _, cover_ratio_dict = self.load_feature_from_dataframe(feature_data, universe=universe,
                                                               timedelta=timedelta, transformer=norm.standard_scale,
                                                               cover_rate=True)
        alpha_name = feature_data.columns[0]
        cover_ratio = cover_ratio_dict[alpha_name]
        if cover_ratio < float(self.options.COVER_RATE_LIMIT):
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}, cover_ratio is too low.")
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason="cover_ratio is too low.")
            return status

        # eval IC
        self.load_return_data(price='close')
        ic_flag, trading_direction = self.test_ic(alpha_name, output=output, verbose=verbose)
        if trading_direction is None:
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}, cannot calculate trading direction.")
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio,
                                          change_reason="cannot calculate trading direction.")
            return status

        if ic_flag:
            corr_flag, names, md5, msg = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name,
                                                                                    verbose=verbose)
            # enter factor_db
            if corr_flag:
                status = 'accepted'
                status_json["status"] = 'accepted_no_replace'
            else:
                if names is None:
                    status_json = json.dumps(status_json)
                    self.dist.to_redis_feature_status(md5, status_json)
                    self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=msg)
                    return status
                else:
                    # compare ic value, correlation eval fail
                    _, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
                    alphas_in_db = dict()
                    for name in names:
                        ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                        return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                        alphas_in_db[name] = dict()
                        alphas_in_db[name]['ic'] = ic_target
                        alphas_in_db[name]['ret'] = return_target
                    status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db,
                                                                                    return_info,
                                                                                    self.ic_table)
                    if status == 'accepted' and replace_reason:
                        self.db_api.update_to_watch(names, replace_reason)
                        status_json["status"] = 'accepted_with_replace'
                        status_json["alpha_name"] = names
                    else:
                        pass

            # set redis flag
            send_flag = True

        else:
            return_flag, return_info = self.test_return(alpha_name, trading_direction, verbose=False, output=output)

            if return_flag:
                corr_flag, names, md5, msg = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name,
                                                                                        verbose=False)

                if corr_flag:
                    status = 'accepted'
                    status_json["status"] = 'accepted_no_replace'
                else:
                    if names is None:
                        status_json = json.dumps(status_json)
                        self.dist.to_redis_feature_status(md5, status_json)
                        self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)
                        return status
                    else:
                        alphas_in_db = dict()
                        for name in names:
                            ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                            return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                            alphas_in_db[name] = dict()
                            alphas_in_db[name]['ic'] = ic_target
                            alphas_in_db[name]['ret'] = return_target
                        status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db,
                                                                                        return_info,
                                                                                        self.ic_table)
                    if status == 'accepted' and replace_reason:
                        self.db_api.update_to_watch(names, replace_reason)
                    status_json["status"] = 'accepted_with_replace'
                    status_json["alpha_name"] = names

                # set redis flag
                send_flag = True

        # update table list
        if status in ('accepted', 'watched',):
            self.update_related_info_from_dataframe_dist(alpha_name, feature_data, trading_direction,
                                                         benchmark='Investable')

        # send redis here
        if send_flag:
            status_json = json.dumps(status_json)
            self.dist.to_redis_feature_status(md5, status_json)
            self.logger.info(
                f'Operation for corr_matrix of {self.feature_name} has been send.')

        # insert raw data into db
        if status in ('accepted', 'watched',):
            df_format = feature_data.reset_index()
            df_format['timestamp'] = df_format['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_format.rename(columns={'timestamp': "trading_time",
                                      'ticker': 'symbol', alpha_name: "alpha_value"}, inplace=True)
            benchmarks = self.options.EVAL_UNIVERSES.copy()
            benchmarks.remove('Investable')
            for bench in benchmarks:
                self.update_related_info_from_dataframe_dist(alpha_name, feature_data, trading_direction,
                                                             benchmark=bench)
            self.db_api.insert_to_db(df_format.dropna(subset=["alpha_value"]), alpha_name)
            self.db_api.register_data_info(alpha_name, feature_data)
            self.db_api.register(researcher, alpha_name, trading_direction, cover_ratio, critical_value,
                                 category=category, status=status, change_reason=change_reason)
        else:
            self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)

        if self.logger is not None:
            self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
                             f"cover_ratio: {round(cover_ratio, 3)}")
        return status

    def distribute_update_feature_from_dataframe(self, researcher, feature_data,
                                                 universe='Investable',
                                                 timedelta=None, transformer=norm.rank, output=None,
                                                 critical_value=None, verbose=False):

        # parameters
        status = 'rejected'
        change_reason = None
        send_flag = False
        status_json = dict()
        status_json["status"] = 'no_action'
        _periods = self.options.EVAL_PERIODS
        df_format = feature_data.reset_index()
        alpha_name = feature_data.columns[0]
        df_format['timestamp'] = df_format['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        df_format.rename(columns={'timestamp': "trading_time",
                                  'ticker': 'symbol', alpha_name: "alpha_value"}, inplace=True)
        self.db_api.insert_to_db(df_format.dropna(subset=["alpha_value"]), alpha_name)
        self.db_api.update_data_info(alpha_name, feature_data)
        if self.logger is not None:
            self.logger.info(f"Daily operation of {alpha_name} finished.")
        # # load eval data and check cover rate
        # _, cover_ratio_dict = self.load_feature_from_dataframe(feature_data, universe=universe,
        #                                                        timedelta=timedelta, transformer=transformer,
        #                                                        cover_rate=True)
        # alpha_name = feature_data.columns[0]
        # cover_ratio = cover_ratio_dict[alpha_name]
        # if cover_ratio < COVER_RATE_LIMIT:
        #     self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
        #                      f"cover_ratio: {round(cover_ratio, 3)}, cover_ratio is too low.")
        #     self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason="cover_ratio is too low.")
        #     return status

        # # eval IC
        # self.load_return_data(price='close')
        # ic_flag, trading_direction = self.test_ic(alpha_name, output=output, verbose=verbose)
        # if trading_direction is None:
        #     self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
        #                      f"cover_ratio: {round(cover_ratio, 3)}, cannot calculate trading direction.")
        #     self.db_api.register_rejected(researcher, alpha_name, cover_ratio,
        #                                   change_reason="cannot calculate trading direction.")
        #     return status

        # if ic_flag:
        #     corr_flag, names, md5 = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name, verbose=verbose)
        #     # enter factor_db
        #     if corr_flag:
        #         status = 'accepted'
        #         status_json["status"] = 'accepted_no_replace'
        #     else:
        #         if names is None:
        #             status_json = json.dumps(status_json)
        #             self.dist.to_redis_feature_status(md5, status_json)
        #             self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)
        #             return status
        #         else:
        #             # compare ic value, correlation eval fail
        #             _, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
        #             alphas_in_db = dict()
        #             for name in names:
        #                 ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
        #                 return_target = self.db_api.query_factor_rec_return(name, period=_periods)
        #                 alphas_in_db[name] = dict()
        #                 alphas_in_db[name]['ic'] = ic_target
        #                 alphas_in_db[name]['ret'] = return_target
        #             status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db, return_info,
        #                                                                        self.ic_table)
        #              if status == 'accepted' and replace_reason:
        #                 self.db_api.update_to_watch(names, replace_reason)
        #                 status_json["status"] = 'accepted_with_replace'
        #                 status_json["alpha_name"] = names
        #             else:
        #                 pass

        #     # set redis flag
        #     send_flag = True

        # else:
        #     return_flag, return_info = self.test_return(alpha_name, trading_direction, verbose=False, output=output)

        #     if return_flag:
        #         corr_flag, names, md5 = self.get_correlations_with_factor_lib_dist(feature_name=alpha_name,
        #                                                                            verbose=False)

        #         if corr_flag:
        #             status = 'accepted'
        #             status_json["status"] = 'accepted_no_replace'
        #         else:
        #             if names is None:
        #                 status_json = json.dumps(status_json)
        #                 self.dist.to_redis_feature_status(md5, status_json)
        #                 self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)
        #                 return status
        #             else:
        #                 alphas_in_db = dict()
        #                 for name in names:
        #                     ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
        #                     return_target = self.db_api.query_factor_rec_return(name, period=_periods)
        #                     alphas_in_db[name] = dict()
        #                     alphas_in_db[name]['ic'] = ic_target
        #                     alphas_in_db[name]['ret'] = return_target
        #                 status, change_reason, msg, replace_reason = get_feature_status(alpha_name, alphas_in_db,
        #                                                                            return_info,
        #                                                                            self.ic_table)
        #              if status == 'accepted' and replace_reason:
        #                 self.db_api.update_to_watch(names, replace_reason)
        #             status_json["status"] = 'accepted_with_replace'
        #             status_json["alpha_name"] = names

        #         # set redis flag
        #         send_flag = True

        # # update table list
        # if status in ('accepted', 'watched',):
        #     self.db_api.register(researcher, alpha_name, trading_direction, cover_ratio, critical_value,
        #                          category=category, status=status, change_reason=change_reason)
        #     self.update_related_info_from_dataframe_dist(alpha_name, feature_data, trading_direction,
        #                                                  benchmark='Investable')
        # else:
        #     self.db_api.register_rejected(researcher, alpha_name, cover_ratio, change_reason=change_reason)

        # # send redis here
        # if send_flag:
        #     status_json = json.dumps(status_json)
        #     self.dist.to_redis_feature_status(md5, status_json)
        #     self.logger.info(
        #         f'Operation for corr_matrix of {self.feature_name} has been send.')

        # # insert raw data into db
        # if status in ('accepted', 'watched',):
        #     df_format = feature_data.reset_index()
        #     df_format['timestamp'] = df_format['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        #     df_format.rename(columns={'timestamp': "trading_time",
        #                               'ticker': 'symbol', alpha_name: "alpha_value"}, inplace=True)
        #     self.db_api.insert_to_db(df_format.dropna(subset=["alpha_value"]), alpha_name)
        #     self.db_api.register_data_info(alpha_name, feature_data)
        #     benchmarks = EVAL_UNIVERSES.copy()
        #     benchmarks.remove('Investable')
        #     for bench in benchmarks:
        #         self.update_related_info_from_dataframe_dist(alpha_name, feature_data, trading_direction,
        #                                                      benchmark=bench)
        # if self.logger is not None:
        #     self.logger.info(f"Test of {alpha_name} finished,  result status: {status}, "
        #                      f"cover_ratio: {round(cover_ratio, 3)}")
        return

    def generate_alpha_daily_ic_from_file(self, alpha_name, universe, price_mode, **kwargs):
        try:
            self.load_return_data(price=price_mode, **kwargs)
            self.get_intersection_ic(feature_name=alpha_name, truncate_fold=None, method='spearman', period=(
                '5m', '15m', '30m', '60m', '120m', '1d', '2d', '3d', '4d', '5d', '10d', '20d'))
            self._handle_daily_ic_info(alpha_name, self.ic_table, universe)
            self.logger.info(f"Inseration daily ic info of {alpha_name} succeed!")
        except Exception as err:
            self.logger.error(f"Inseration daily ic info of {alpha_name} fails, error message: {err}")
            print(traceback.format_exc())

    def generate_alpha_daily_ic_from_dataframe(self, universe, price_mode, **kwargs):
        pass

    def generate_alpha_daily_return(self, alpha_name, universe, negative, group_num=5, price_mode='close', **kwargs):
        try:
            df, _, _ = self.get_group_returns(alpha_name, negative=negative, group_num=group_num, **kwargs)
            # may change
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d').astype(int)

            # already sorted by standard file
            df.set_index('timestamp', inplace=True)

            bt = BTFeatureDaily(configs, start_date=FT.trading_days[0], end_date=FT.trading_days[-1],
                                logger=FT.logger)
            bt.feed_data(df)
            result = bt.run_group_test(group=group_num)
            time_list = [930, 935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045,
                         1050, 1055, 1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330,
                         1335, 1340, 1345, 1350, 1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445,
                         1450, 1455, 1500]
            time_target = time_list[int(self.options.ti)]
            result.index = (pd.to_datetime(result.index, format='%Y%m%d') +
                            pd.Timedelta(hours=time_target // 100, minutes=time_target % 100)). \
                strftime('%Y-%m-%d %H:%M:%S')

            # change here!
            # print(result.head())
            # quit()
            result.reset_index(inplace=True)
            result.rename(columns={"timestamp": "trading_time"}, inplace=True)
            result["alpha_name"] = alpha_name
            self.db_api.insert_to_ret_group_daily(result, price_mode, group=group_num, throw_dupl=False)

            self.logger.info(f"Test of {alpha_name} succeed!")
        except Exception as err:
            self.logger.error(f"Inseration daily return info of {alpha_name} fails, error message: {err}")
            print(traceback.format_exc())
