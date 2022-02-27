# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
# version: 2022.02.10
import calendar
import datetime
import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd
from DataAPI import get_trading_days, get_universe, get_last_trading_day, convert_to_datetime
from tabulate import tabulate
from tqdm import tqdm

from ..backtest.bt import BTFeatureDaily
from ..config.config import COVER_RATE_LIMIT, BT_STOCK_PCT, FACTOR_LIB_MYSQL_TIO, BT_TAX_RATE, BT_TRANSMISSION_RATE, \
    EVAL_PERIODS
from ..database.mysql import MysqlAPI
from ..metrics.entrance_requirements import is_correlation_accepted, is_ic_ir_accepted, is_return_accepted, \
    get_feature_status
from ..utils import normalization as norm

warnings.filterwarnings('ignore')


def time_5m_calculator(timestamp, ti):
    assert isinstance(ti, int) and ti >= 0, "ti should be int greater than 0!"
    time_list = [930, 935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045,
                 1050, 1055, 1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330,
                 1335, 1340, 1345, 1350, 1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445,
                 1450, 1455, 1500]
    hour = timestamp.hour
    minute = timestamp.minute
    time_target = int(str(hour).zfill(2) + str(minute).zfill(2))
    ti_start = time_list.index(time_target)
    ti_sum = ti_start + ti
    days = ti_sum // 49
    time_end = time_list[ti_sum % 49]
    time_target_final = datetime.datetime.fromordinal(
        get_trading_days(start_date=timestamp.date(), count=days + 1, output='datetime')[-1].toordinal()) + \
                        datetime.timedelta(hours=time_end // 100, minutes=time_end % 100)
    return time_target_final


class Reader:
    @staticmethod
    def csv_reader(input_set, transformer=None, missing_file_allowed=False, alpha_mode=True, grouped=True):
        path, alpha_name, date = input_set
        try:
            df = pd.read_csv(path, header=None, names=['timestamp', 'ticker', alpha_name], dtype={1: str})
        except FileNotFoundError:
            if missing_file_allowed:
                return date, alpha_name, None
            else:
                raise FileNotFoundError(f"File {path} not found!")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index(['timestamp', 'ticker'], inplace=True)
        if not alpha_mode:
            return date, alpha_name, df

        if transformer is not None:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])
            df[alpha_name], _, _ = norm.handle_extreme_value(df[alpha_name].values, method='MAD')
            df[alpha_name] = transformer(df[alpha_name])
        else:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])

        if grouped:
            grouped_df = df.groupby(level=0, sort=False)
            result = [(timestamp, group.droplevel(0)) for timestamp, group in grouped_df]
            return date, alpha_name, result
        else:
            return date, alpha_name, df

    @staticmethod
    def par_reader(input_set, transformer=None, missing_file_allowed=False, alpha_mode=True, grouped=True,
                   filter_trading_status=None):
        path, alpha_name, date = input_set
        try:
            df = pd.read_parquet(path)
        except FileNotFoundError:
            if missing_file_allowed:
                return date, alpha_name, None
            else:
                raise FileNotFoundError(f"File {path} not found!")
        if not alpha_mode:
            if filter_trading_status is not None:
                df = df.loc[~df['trading_status'].isin(filter_trading_status)]
            if transformer is not None:
                columns = df.columns.tolist()
                columns.remove('trading_status')
                df.loc[:, columns] = (df.loc[:, columns].apply(transformer, axis=0))
            return date, alpha_name, df

        if transformer is not None:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])
            df[alpha_name], _, _ = norm.handle_extreme_value(df[alpha_name].values, method='MAD')
            df[alpha_name] = transformer(df[alpha_name])
        else:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])

        if grouped:
            grouped_df = df.groupby(level=0, sort=False)
            result = [(timestamp, group.droplevel(0)) for timestamp, group in grouped_df]
            return date, alpha_name, result
        else:
            return date, alpha_name, df

    @staticmethod
    def dataframe_reader(input_set, transformer=None, missing_file_allowed=False):
        timestamp, df = input_set
        columns = df.columns.tolist()

        def _convert(ser):
            temp = ser.loc[~ser.isna()]
            temp = norm.handle_extreme_value(temp, method='MAD')[0]
            temp = transformer(temp)
            return temp

        if transformer is not None:
            df.mask(np.isinf(df), inplace=True)

            for item in columns:
                series = df[item]
                df.loc[~series.isna(), item] = _convert(series)

                if not missing_file_allowed:
                    if df[item].dropna().empty:
                        raise ValueError(f"Data: {item} - {timestamp} is missing!")
        # df.loc[:, columns] = (df.loc[:, columns].apply(lambda x: norm.handle_extreme_value(x, method='MAD')[0], axis=0))
        # df.loc[:, columns] = (df.loc[:, columns].apply(transformer, axis=0))
        else:
            df.mask(np.isinf(df), inplace=True)
        return timestamp, df


class FeatureAnalysis(object):
    def __init__(self, options, mysql_info=FACTOR_LIB_MYSQL_TIO, feature_path=None, logger=None,
                 workers=mp.cpu_count()):

        self.options = options
        self.workers = workers
        mysql_info['db'] = self.options.db
        self.db_api = MysqlAPI(mysql_info=mysql_info, logger=None)
        self.logger = logger
        self.ti = self.options.ti
        if feature_path is not None:
            self.pathway = os.path.join(feature_path, "ti" + str(self.ti))
            self.features_in_path = self.get_features_in_path()
        else:
            self.pathway = None
            self.features_in_path = None

        self.corr_path = os.path.join(options.corr_path, options.db, "corr", "corr.par")
        self.corr_matrix_loaded = False

        if options.freq == 'daily':
            self.return_path = os.path.join(options.return_path_daily, '5min')
        else:
            self.return_path = os.path.join(options.return_path_minute, options.freq)

        try:
            self.db_features_info = self.db_api.get_factor_info()
            self.features_in_db = self.db_features_info['table_name'].values.tolist()
        except pd.io.sql.DatabaseError:
            self.db_features_info = pd.DataFrame()
            self.features_in_db = []

        self.feature_data = None
        self.target_data = None
        self.data_info = None

        self.trading_days = None
        self.feature_names = None
        self.feature_name = None

        self.ic_table = None
        self.trading_direction = None
        self.time_delta = None
        self.return_table = None
        self.corr_table = None

        self.mic_table = None
        self.VIF_table = None
        self.mic_target = None
        self.ic_truncated = None
        self.time_list = [930, 935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045,
                          1050, 1055, 1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330,
                          1335, 1340, 1345, 1350, 1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445,
                          1450, 1455, 1500]
        self.stock_num_count = None

        self.corr_matrix = None
        self.corr_end_time = None
        self.corr_start_time = None

        self.pool = mp.Pool(processes=workers)

    def get_features_in_path(self):
        if self.pathway is not None:
            researcher_list = os.listdir(self.pathway)
            factors_dict = dict()
            for researcher in researcher_list:
                factors = os.listdir(os.path.join(self.pathway, researcher))
                for factor in factors:
                    factors_dict[factor] = researcher
            return factors_dict
        else:
            return None

    def load_corr_matrix(self):
        try:
            self.corr_matrix = pd.read_parquet(self.corr_path)
            self.corr_end_time = self.corr_matrix.index.get_level_values(0).max()
            self.corr_start_time = self.corr_matrix.index.get_level_values(0).min()
        except FileNotFoundError:
            pass
        self.corr_matrix_loaded = True

    def load_feature_from_file(self, feature_names, start_time, end_time, universe='Investable', timedelta=None,
                               transformer=None, cover_rate=False, check_timestamp=False):
        # init
        self.reset()
        assert self.pathway is not None, \
            "if you want to upload from file, please assign initial parameter `feature_path`!"
        missing_file_allowed = self.options.daily_data_missing_allowed
        if timedelta is not None:
            self.time_delta = timedelta
        if isinstance(feature_names, str):
            feature_names = [feature_names]

        # time range
        start_time = convert_to_datetime(start_time)
        end_time = convert_to_datetime(end_time)
        if end_time < \
                datetime.datetime.fromordinal(end_time.date().toordinal()) + datetime.timedelta(hours=9, minutes=30):
            trade_days = get_trading_days(start_time.date(), get_last_trading_day(end_time.date()), output='datetime')
        else:
            trade_days = get_trading_days(start_time.date(), end_time.date(), output='datetime')

        handler = getattr(Reader, f"{self.options.feature_data_format}_reader")
        self.trading_days = trade_days
        self.feature_names = feature_names

        # load files
        feature_path_list = []
        for feature_name in feature_names:
            feature_path_list.extend(
                [(os.path.join(self.pathway, self.features_in_path[feature_name],
                               feature_name, str(item.year), datetime.datetime.strftime(item, '%Y%m%d')
                               + '.' + self.options.feature_data_format), feature_name, item) for item in trade_days])

        cover_rate_value = None
        if universe == 'All' and cover_rate is False:
            reader = FeatureAnalysis.read_files
            self.feature_data = reader(feature_path_list, pool=self.pool,
                                       scope='Loading Feature...',
                                       handler=handler, time_delta=timedelta, transformer=transformer,
                                       missing_file_allowed=missing_file_allowed)
        else:
            reader = FeatureAnalysis.read_files_filter
            self.feature_data, cover_rate_value = reader(feature_path_list, universe=universe, pool=self.pool,
                                                         scope='Loading Feature...',
                                                         handler=handler, time_delta=timedelta, transformer=transformer,
                                                         cover_rate=cover_rate,
                                                         missing_file_allowed=missing_file_allowed)
        self.feature_data = \
            self.feature_data.loc[(self.feature_data.index.get_level_values(0) >= pd.Timestamp(start_time)) &
                                  (self.feature_data.index.get_level_values(0) <= pd.Timestamp(end_time))]

        if timedelta is not None:
            self.ti += self.time_delta

        # check timestamp
        if check_timestamp:
            time_list = self.feature_data.index.get_level_values(0).unique()
            time_list = list(set(zip(time_list.hour, time_list.minute)))
            assert len(time_list) == 1, f"Feature data generate at a different time, {time_list}"
            time_target = int(str(time_list[0][0]).zfill(2) + str(time_list[0][1]).zfill(2))
            ti = self.time_list.index(time_target)
            assert self.ti == ti, \
                f"Parameter `ti` in config file is different from `ti` in alpha files, {self.ti} vs {ti}"

        return self.feature_data, cover_rate_value

    # def load_feature_data_from_db(self, feature_name, start_time, end_time, universe='Investable'
    #                               , timedelta=None, normalized=True, norm_method=norm.standard_scale,
    #                               missing_file_allowed=True, cover_rate=False):
    #
    #     start_time = convert_to_datetime(start_time)
    #     end_time = convert_to_datetime(end_time)
    #
    #     df = self.db_api.query_factors(feature_name, start_time=start_time, end_time=end_time)
    #     df = df.reindex(columns=['alpha_value'])
    #     df.rename(columns={'alpha_value': feature_name}, inplace=True)
    #
    #     target_days = df.index.get_level_values(0).unique().strftime('%Y-%m-%d').tolist()
    #     self.trading_days = get_trading_days(start_date=min(target_days), end_date=max(target_days))
    #     if not missing_file_allowed:
    #         for date in self.trading_days:
    #             if str(date) not in target_days:
    #                 raise FileNotFoundError(f"Data of {date} not found!")
    #
    #     handler = getattr(Reader, "dataframe_reader")
    #     self.feature_names = feature_name
    #
    #     grouped = df.groupby(level=0)
    #     feature_path_list = [(timestamp, group, feature_name,) for timestamp, group in grouped]
    #     cover_rate_value = None
    #     if universe == 'All' and cover_rate is False:
    #         reader = FeatureAnalysis.db_reader
    #         self.feature_data = reader(feature_path_list, self.pool,
    #                                    scope='Loading Feature {}...'.format(feature_name.lower()),
    #                                    handler=handler, time_delta=timedelta, normalized=normalized,
    #                                    norm_method=norm_method)
    #     else:
    #         reader = FeatureAnalysis.db_reader_filter
    #         self.feature_data, cover_rate_value = reader(feature_path_list, universe=universe, pool=self.pool,
    #                                                      scope='Loading Feature {}...'.format(feature_name.lower()),
    #                                                      handler=handler, time_delta=timedelta,
    #                                                      normalized=normalized, norm_method=norm_method,
    #                                                      cover_rate=cover_rate)
    #     self.feature_data.rename_axis(index={'symbol': 'ticker', 'trading_time': 'timestamp'}, inplace=True)
    #     time_target = self.feature_data[list(self.feature_data.keys())[0]].index[0][0]
    #     hour = time_target.hour
    #     minute = time_target.minute
    #     time_target = int(str(hour).zfill(2) + str(minute).zfill(2))
    #     self.ti = self.time_list.index(time_target)
    #
    #     return self.feature_data, cover_rate_value

    #

    def load_feature_from_dataframe(self, feature_data, universe='Investable', timedelta=None, transformer=None,
                                    cover_rate=False, check_timestamp=False):
        self.reset()
        assert feature_data.index.nlevels == 2, "feature data should be dataframe with indexes (`timestamp, `ticker)"

        if check_timestamp:
            time_list = feature_data.index.get_level_values(0).unique()
            time_list = list(set(zip(time_list.hour, time_list.minute)))
            assert len(time_list) == 1, f"Feature data generate at a different time, {time_list}"
            time_target = int(str(time_list[0][0]).zfill(2) + str(time_list[0][1]).zfill(2))
            ti = self.time_list.index(time_target)
            assert self.ti == ti, \
                f"Parameter `ti` in config file is different from `ti` in alpha files, {self.ti} vs {ti}"
        if timedelta is not None:
            self.time_delta = timedelta
        if check_timestamp:
            missing_file_allowed = self.options.daily_data_missing_allowed
        else:
            missing_file_allowed = True

        feature_names = feature_data.columns.tolist()

        target_times = feature_data.index.get_level_values(0).unique()
        target_days_min = target_times.min().strftime('%Y-%m-%d')
        target_days_max = target_times.max().strftime('%Y-%m-%d')
        self.trading_days = get_trading_days(start_date=target_days_min, end_date=target_days_max)
        feature_data = feature_data.dropna(how='all')

        if not missing_file_allowed:
            target_times = feature_data.index.get_level_values(0).unique()
            target_days = [item.date() for item in target_times]
            for item in self.trading_days:
                assert item in target_days, f'Data: {item} of input feature is missing.'

        handler = getattr(Reader, "dataframe_reader")
        self.feature_names = feature_names

        grouped = feature_data.groupby(level=0)
        feature_path_list = [(timestamp, group.droplevel(0)) for timestamp, group in grouped]

        cover_rate_value = None
        if universe == 'All' and cover_rate is False:
            reader = FeatureAnalysis.db_reader
            self.feature_data = reader(feature_path_list, self.pool, scope='Loading Feature...',
                                       handler=handler, time_delta=timedelta, transformer=transformer,
                                       missing_file_allowed=missing_file_allowed)
        else:
            reader = FeatureAnalysis.db_reader_filter
            self.feature_data, cover_rate_value = reader(feature_path_list, universe=universe, pool=self.pool,
                                                         scope='Loading Feature...',
                                                         handler=handler, time_delta=timedelta, transformer=transformer,
                                                         cover_rate=cover_rate,
                                                         missing_file_allowed=missing_file_allowed)
        self.feature_data.rename_axis(index={'symbol': 'ticker', 'trading_time': 'timestamp'}, inplace=True)

        if timedelta is not None:
            self.ti += self.time_delta
        return self.feature_data, cover_rate_value

    def load_normalized_feature(self, feature_data, check_timestamp=False):
        self.reset()
        self.feature_data = feature_data
        self.feature_names = self.feature_data.columns.tolist()
        target_days = self.feature_data.index.get_level_values(0).unique().strftime('%Y-%m-%d').tolist()
        self.trading_days = get_trading_days(start_date=min(target_days), end_date=max(target_days))

        # check timestamp
        if check_timestamp:
            time_list = self.feature_data[list(self.feature_data.keys())[0]].index.get_level_values(0).unique()
            time_list = list(zip(time_list.hour, time_list.minute))
            assert len(time_list) == 1, f"Feature data generate at a different time, {time_list}"
            time_target = int(str(time_list[0][0]).zfill(2) + str(time_list[0][1]).zfill(2))
            ti = self.time_list.index(time_target)
            assert self.ti == ti, \
                f"Parameter `ti` in config file is different from `ti` in alpha files, {self.ti} vs {ti}"

    def load_return_data(self, filter_trading_status=(1, 2, 3), tp=0, price='close', transformer=None):
        """
        :param tp:
        :param price:
        :param filter_trading_status:
        :param transformer:
        :return:
        """
        assert self.ti is not None, f"feature data must be set at first!"

        path_base = self.return_path
        feature_path_list = \
            [(os.path.join(path_base, price, 'ti' + str(self.ti), 'tp' + str(tp), str(item.year),
                           '{:0>2}'.format(item.month), datetime.datetime.strftime(item, '%Y%m%d') + '.par'), None,
              item)
             for item in self.trading_days]
        self.target_data = FeatureAnalysis.read_files_target(feature_path_list, self.pool,
                                                             scope='Loading target...', transformer=transformer,
                                                             handler=Reader.par_reader,
                                                             filter_trading_status=filter_trading_status)
        self.data_info = dict()
        stock_num_count = []
        grouped_target = self.target_data.groupby(level=0, sort=False)
        grouped_feature = self.feature_data.groupby(level=0, sort=False)
        with tqdm(total=grouped_target.ngroups, ncols=150) as pbar:
            for timestamp, group in grouped_target:
                group = group.reset_index(level=0, drop=True)
                try:
                    group_target = grouped_feature.get_group(timestamp).reset_index(level=0, drop=True)
                except KeyError:
                    if self.logger:
                        self.logger.warn(
                            f'Timestamp:{timestamp} of feature {self.feature_names} has not available stock.')
                    pbar.set_description("Merging feature and return...")
                    pbar.update(1)
                    continue
                temp = group_target.join(group, how='inner')
                self.data_info[timestamp] = temp
                stock_num_count.append(temp.shape[0])

                pbar.set_description("Merging feature and return...")
                pbar.update(1)

        self.stock_num_count = np.mean(stock_num_count)

    def get_intersection_ic(self, feature_name,
                            truncate_fold=None,
                            period=('5m', '15m', '30m', '60m', '120m', '1d', '2d', '3d', '4d', '5d'),
                            method='pearson'):
        """
        :param feature_name:
        :param truncate_fold:
        :param period:
        :param method: {'pearson', 'kendall', 'spearman'} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays
        """
        assert self.data_info is not None, "features and target should be loaded!"
        self.feature_name = feature_name
        if isinstance(period, str):
            period = [period]

        columns = [feature_name]
        for period_ in period:
            columns.append(period_ + "_ret")

        target_info = [(key, self.data_info[key]) for key in self.data_info.keys()]

        stock_iter = iter(target_info)
        pool = self.pool

        # 初始化任务
        result_list = [pool.apply_async(FeatureAnalysis._calculate_ic,
                                        args=(next(stock_iter), columns, truncate_fold, method,))
                       for _ in range(min(self.pool._processes, len(self.data_info)))]

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
                        pbar.set_description("Calculating IC value...")
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(
                                    pool.apply_async(FeatureAnalysis._calculate_ic,
                                                     args=(next(stock_iter), columns, truncate_fold, method,)))
                            except StopIteration:
                                flag = 0

        column_final = ['timestamp']
        for target in columns[1:]:
            column_final.append('IC_' + feature_name + '_' + target.split('_')[0])

        if truncate_fold is not None:
            for target in columns[1:]:
                column_final.append('ICT_' + feature_name + '_' + target.split('_')[0])
        self.ic_table = pd.DataFrame(df_list, columns=column_final)

        self.ic_table.set_index(keys='timestamp', inplace=True)
        self.ic_table.sort_index(inplace=True)

    def get_ic_summary_by_month(self, num=1, total_table=True, verbose=True):
        """
        :param num:
        :param total_table:
        :param verbose:
        :return:
        """
        df = self.ic_table
        df.index = pd.to_datetime(df.index.astype(str), format='%Y-%m-%d %H:%M:%S')
        date_list = df.index.tolist()
        date_start, date_end = date_list[0], date_list[-1]

        # IC_ALPHA_5m
        name = ['_'.join(item.split('_')[0:-2][0:2]) +
                '_' + item.split('_')[-1] for item in df.columns.tolist()]
        df.columns = name

        _, date_end_flag = self._get_date_to_slice(date_start, num=num)
        date_end_flag += datetime.timedelta(hours=15)
        result_list = []
        flag = 1
        while date_end_flag <= date_end and flag == 1:
            if date_end_flag == date_end:
                flag = 0
                next_start = None
            else:
                next_start, date_end_flag = self._get_date_to_slice(date_start, num=num)

            df_temp = df[(df.index <= date_end_flag) & (df.index >= date_start)]
            temp_list = []
            for item in name:
                temp_list.append('{:.2f}'.format(df_temp[item].mean()) + ' ' + '{:.2f}'.format(df_temp[item].std()) +
                                 ' ' + '{:.2f}'.format(df_temp[item].mean() / df_temp[item].std()))
            temp_list.insert(0, '{:8s}-{:8s}'.format(datetime.datetime.strftime(date_start, '%Y%m%d'),
                                                     datetime.datetime.strftime(date_end_flag, '%Y%m%d')))
            result_list.append(temp_list)

            date_start = next_start
            date_end_flag = date_end_flag + datetime.timedelta(days=1)

        total = []
        for item in name:
            total.append('{:.2f}'.format(df[item].mean()) + ' ' + '{:.2f}'.format(df[item].std()) +
                         ' ' + '{:.2f}'.format(df[item].mean() / df[item].std()))
        total.insert(0, '{:8s}-{:8s}'.format(datetime.datetime.strftime(date_list[0], '%Y%m%d'),
                                             datetime.datetime.strftime(date_list[-1], '%Y%m%d')))
        if not result_list:
            result_list = [total]

        columns = ['period']
        for item in name:
            columns.append(item)

        ic_table = pd.DataFrame(result_list, columns=columns)
        ic_table.set_index('period', inplace=True)

        total_df = pd.DataFrame([total], columns=columns)
        total_df.set_index('period', inplace=True)

        if verbose:
            print('\nIC table for feature "{}" (mean std IR):'.format(self.feature_name))
            print(tabulate(ic_table, headers=['period'] + ic_table.columns.tolist(),
                           tablefmt='grid', floatfmt=".4f", stralign="center", numalign="center"))
            if total_table:
                print('\nIC summary for feature "{}":'.format(self.feature_name))
                print(tabulate(total_df, headers=['period'] + total_df.columns.tolist(),
                               tablefmt='grid', floatfmt=".4f", stralign="center", numalign="center"))

        return ic_table, total_df

    def get_top_return(self, feature_name, negative=False, trade_type='long-short', stock_pct=0.1,
                       transmission_rate=0.0,
                       tax_rate=0.0, period=6, weight='equal', verbose=False, **kwargs):
        """
        :param feature_name:
        :param negative:
        :param trade_type: {'long-only', 'short-only', 'long-short'}
                           long-only : only execute long direction
                           short-only : only execute short direction
                           long-short : execute long & short direction
        :param transmission_rate:
        :param stock_pct:
        :param tax_rate:
        :param period:
        :param weight:
        :param verbose:
        :return: (pd.DataFrame, pd.DataFrame)
        """
        assert self.feature_data is not None, "features data should be loaded!"
        self.feature_name = feature_name
        configs = self.options.copy()

        df = self.feature_data.reindex(columns=[feature_name]).reset_index()
        if negative:
            df[feature_name] = -1 * df[feature_name]

        # may change
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d').astype(int)

        # already sorted by standard file
        df.set_index('timestamp', inplace=True)

        # configs.trade_period = 0
        configs.ti = self.ti
        for arg in kwargs.keys():
            if arg in ("score_sorted", "daily_data_missing_allowed", "bt_price", "ti", "trade_period",
                       "constant_trading_stock_num"):
                setattr(configs, arg, kwargs[arg])
        if configs.bt_price in ('twap', "vwap"):
            assert configs.trade_period != 0, "When config.bt_price is `twap` or `vwap`, `trade_period` cannot be 0."
        configs.weight = weight
        configs.transmission_rate = transmission_rate
        configs.tax_rate = tax_rate
        configs.stock_percentage = True
        configs.stock_num = stock_pct
        configs.constant_trading_stock_num = True

        # may change
        configs.trading_type = trade_type

        bt = BTFeatureDaily(configs, start_date=self.trading_days[0], end_date=self.trading_days[-1],
                            logger=self.logger)
        bt.feed_data(df)
        # bt.run_feature(start_num=start_num, end_num=end_num, mode=mode)
        bt.run()

        if verbose:
            print('\n{} back-test result for {} mode:'.format(self.feature_name, trade_type))
        date_start, date_end = datetime.datetime.fromordinal(self.trading_days[0].toordinal()), \
                               datetime.datetime.fromordinal(self.trading_days[-1].toordinal())
        _, date_end_flag = self._get_date_to_slice(date_start, num=period)
        result_list = []
        flag = 1
        while date_end_flag <= date_end and flag == 1:
            next_start, date_end_flag = self._get_date_to_slice(date_start, num=period)
            if date_end_flag > date_end:
                date_end_flag = date_end
                flag = 0
            elif date_end_flag == date_end:
                flag = 0
            result_list.append((int(date_start.strftime('%Y%m%d')), int(date_end_flag.strftime('%Y%m%d'))))
            date_start = next_start

        self.return_table = dict()
        if result_list:
            df_return, df_return_summary = bt.evaluate(evalRange=tuple(result_list), verbose=verbose)
            self.return_table[trade_type] = df_return
            return df_return, df_return_summary, bt
        else:
            df_return, df_return_summary = bt.evaluate(evalRange=None, verbose=verbose)
            self.return_table[trade_type] = df_return
            return df_return, df_return_summary, bt

    def get_group_returns(self, feature_name, negative=False, group_num=5, transmission_rate=0.0,
                          tax_rate=0.0, **kwargs):
        assert self.feature_data is not None, "features data should be loaded!"
        df = self.feature_data.reindex(columns=[feature_name]).reset_index()
        configs = self.options.copy()

        # configs.trade_period = 0
        # configs.bt_price = 'close'
        configs.ti = self.ti
        for arg in kwargs.keys():
            if arg in ("score_sorted", "daily_data_missing_allowed", "bt_price", "ti", "trade_period",
                       "constant_trading_stock_num", "benchmark"):
                setattr(configs, arg, kwargs[arg])
        if configs.bt_price in ('twap', "vwap"):
            assert configs.trade_period != 0, "When config.bt_price is `twap` or `vwap`, `trade_period` cannot be 0."
        configs.transmission_rate = transmission_rate
        configs.tax_rate = tax_rate
        configs.stock_percentage = True
        configs.constant_trading_stock_num = True

        if negative:
            df[feature_name] = -1 * df[feature_name]

        # may change
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d').astype(int)

        # already sorted by standard file
        df.set_index('timestamp', inplace=True)

        bt = BTFeatureDaily(configs, start_date=self.trading_days[0], end_date=self.trading_days[-1],
                            logger=self.logger)
        bt.feed_data(df)
        result, holdings = bt.run_group_test(group=group_num, tax_rate=tax_rate, transmission_rate=transmission_rate)
        return result, holdings

    def get_correlations_with_factor_lib(self, feature_name=None, method='spearman'):
        """
        :param
        method: {'pearson', 'kendall', 'spearman'} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays
        """
        if not self.corr_matrix_loaded:
            self.load_corr_matrix()

        if self.corr_matrix is None:
            return pd.DataFrame()

        if feature_name is None:
            assert self.feature_name is not None, "feature name has not been set!"
            feature_name = self.feature_name
        assert self.feature_data is not None, "feature data must be set at first!"

        corr_dict = dict()
        stock_num_count = []
        if feature_name in self.corr_matrix.columns:
            if self.logger is not None:
                self.logger.warning(
                    f'{self.feature_names} already in feature db!')
            corr_matrix = self.corr_matrix.drop(columns=[feature_name])
            if corr_matrix.empty:
                return pd.DataFrame()
        else:
            corr_matrix = self.corr_matrix.copy()

        grouped_target = corr_matrix.groupby(level=0, sort=False)
        grouped_feature = self.feature_data.reindex(columns=[feature_name]).groupby(level=0, sort=False)
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
        pool = self.pool

        # 初始化任务
        result_list = [pool.apply_async(FeatureAnalysis._calculate_ic,
                                        args=(next(stock_iter), columns, None, method,))
                       for _ in range(min(self.pool._processes, len(target_info)))]

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
        return corr_table_result.dropna()

    def get_correlation_within_features(self, feature_name, start_time, end_time, others=None, method='spearman'):
        """
        :param
        method: {'pearson', 'kendall', 'spearman'} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarrays
        """

        assert self.feature_data is not None, "feature data must be set at first!"
        assert feature_name in self.feature_data.columns, f"{feature_name} is not in feature data!"

        if others is None:
            others = [item for item in self.feature_names if item != feature_name]
            if not others:
                print("No other feature in feature data!")
                return
        else:
            if isinstance(others, str):
                assert others in self.feature_data.columns, f"{others} is not in feature data!"
                others = [others]
            else:
                for item in others:
                    assert item in self.feature_data.columns, f"{item} is not in feature data!"
        feature_data = self.feature_data.loc[
            (self.feature_data.index.get_level_values(0) <= pd.Timestamp(end_time)) &
            (self.feature_data.index.get_level_values(0) >= pd.Timestamp(start_time))]

        corr_dict = dict()
        stock_num_count = []
        if feature_name in others:
            raise ValueError(f"{feature_name} cannot be in others!")
        else:
            corr_matrix = feature_data.reindex(columns=others)
        grouped_target = corr_matrix.groupby(level=0, sort=False)
        grouped_feature = feature_data.reindex(columns=[feature_name]).groupby(level=0, sort=False)
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
        pool = self.pool

        # 初始化任务
        result_list = [pool.apply_async(FeatureAnalysis._calculate_ic,
                                        args=(next(stock_iter), columns, None, method,))
                       for _ in range(min(self.pool._processes, len(target_info)))]

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
        return corr_table_result.dropna()

    def test_ic(self, feature_name, verbose=True, output=None):
        self.get_intersection_ic(feature_name=feature_name, truncate_fold=None, method='spearman',
                                 period=('1d', '2d', '3d', '4d', '5d'))
        ic_flag, msg, trading_direction = is_ic_ir_accepted(feature_name, self.ic_table)

        if self.logger is not None:
            self.logger.info(msg)
        if verbose:
            print(msg)
        if output is not None:
            df, df_all = self.get_ic_summary_by_month(num=6, verbose=False)
            output = os.path.join(output, self.feature_name)
            if not os.path.exists(output): os.makedirs(output)
            output_file = os.path.join(output, 'ic_table_{}.csv'.format(self.feature_name))
            output_file_all = os.path.join(output, 'ic_summary_{}.csv'.format(self.feature_name))
            df.to_csv(output_file, sep=',', mode='w', header=True, index=True, encoding='utf-8')
            df_all.to_csv(output_file_all, sep=',', mode='w', header=True, index=True, encoding='utf-8')
        return ic_flag, trading_direction

    def test_correlation(self, feature_name, method='spearman', verbose=True, output=None):
        """
        test benchmark : 0.8
        """
        corr_table_result = self.get_correlations_with_factor_lib(method=method)
        flag, msg, names = is_correlation_accepted(feature_name, corr_table_result)

        if flag == 0:
            if self.logger is not None:
                self.logger.warning(msg)
            if verbose:
                print(msg)
            return False, names

        elif flag == 1:
            if self.logger is not None:
                self.logger.info(msg)
            if verbose:
                print(msg)
            return True, names

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
                return True, names

            else:
                if self.logger is not None:
                    self.logger.info(msg)
                if verbose:
                    print(msg)
                return False, names

    def test_return(self, feature_name, trading_direction, verbose=True, output=None):
        if trading_direction == -1:
            negative = True
        else:
            negative = False
        return_daily, df_final, _ = self.get_top_return(feature_name, negative=negative, trade_type='long-short',
                                                        stock_pct=BT_STOCK_PCT,
                                                        transmission_rate=BT_TRANSMISSION_RATE,
                                                        tax_rate=BT_TAX_RATE, period=6, weight='equal',
                                                        verbose=False)
        return_flag, msg, _return = is_return_accepted(feature_name, return_daily)
        if self.logger is not None:
            self.logger.info(msg)
        if verbose:
            print(msg)

        if output is not None:
            path_out = os.path.join(output, feature_name)
            if not os.path.exists(path_out): os.makedirs(path_out)
            path_out_file = os.path.join(path_out, 'return_summary_{}.csv'.format(feature_name))
            df_final.to_csv(path_out_file, sep=',', mode='w', header=True, index=True, encoding='utf-8')
        return return_flag, _return

    def evaluate_feature_from_file(self, alpha_name, start_date, end_date, universe='Investable',
                                   transformer=norm.standard_scale, verbose=True, output=None):
        assert self.pathway is not None, \
            "if you want to upload from file, please assign initial parameter `feature_path`!"
        assert alpha_name not in self.features_in_db, \
            f"{alpha_name} has already been in factor lib! check the name of feature!"
        status = 'rejected'
        change_reason = ""
        msg = ""
        _periods = EVAL_PERIODS

        _, cover_rate = self.load_feature_from_file(alpha_name, start_date, end_date, universe=universe,
                                                    transformer=transformer, cover_rate=True, check_timestamp=True)
        if cover_rate[alpha_name] < COVER_RATE_LIMIT:
            change_reason = f"{alpha_name}'s cover rate is too low and bas been rejected. cover rate:{cover_rate}"
            msg = change_reason
            return 'rejected', change_reason, msg

        self.load_return_data()
        self.feature_name = alpha_name

        ic_flag, trading_direction = self.test_ic(alpha_name, output=output)
        if trading_direction is None:
            change_reason = f"{alpha_name} has no clear direction and bas been rejected."
            msg = change_reason
            return 'rejected', change_reason, msg

        if ic_flag:
            corr_flag, names = self.test_correlation(feature_name=alpha_name, method='spearman', verbose=verbose,
                                                     output=output)
            if corr_flag:
                status = 'accepted'
                if names is None:
                    change_reason = f'With no feature in factor lib, ' \
                                    f'{alpha_name} passes all test and has been accepted.'
                else:
                    change_reason = f'{alpha_name} passes all the test and has been accepted.'
                msg = change_reason
            else:
                # compare ic value, correlation test fail
                if names is None:
                    change_reason = f"{alpha_name} has already in factor lib, check your name or rename your alpha!"
                    msg = change_reason
                    return 'rejected', change_reason, msg
                _, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
                alphas_in_db = dict()
                for name in names:
                    ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                    return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                    alphas_in_db[name] = dict()
                    alphas_in_db[name]['ic'] = ic_target
                    alphas_in_db[name]['ret'] = return_target
                status, change_reason, msg, _ = get_feature_status(alpha_name, alphas_in_db, return_info, self.ic_table)

        else:
            return_flag, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
            if return_flag:
                corr_flag, names = self.test_correlation(feature_name=alpha_name, method='spearman', verbose=verbose,
                                                         output=output)
                if corr_flag:
                    status = 'accepted'
                    if names is None:
                        change_reason = f'With no feature in factor lib, ' \
                                        f'{alpha_name} passes all test and has been accepted.'
                    else:
                        change_reason = f'{alpha_name} passes all the test and has been accepted.'
                    msg = change_reason
                else:
                    alphas_in_db = dict()
                    for name in names:
                        ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                        return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                        alphas_in_db[name] = dict()
                        alphas_in_db[name]['ic'] = ic_target
                        alphas_in_db[name]['ret'] = return_target
                    status, change_reason, msg, _ = get_feature_status(alpha_name, alphas_in_db, return_info,
                                                                       self.ic_table)

        if self.logger is not None:
            self.logger.info(msg)
        if verbose:
            print(msg)

        return status, change_reason, msg

    def evaluate_feature_from_dataframe(self, alpha_name, feature_data, universe='Investable',
                                        transformer=norm.standard_scale, verbose=True, output=None):
        assert self.pathway is not None, \
            "if you want to upload from file, please assign initial parameter `feature_path`!"
        assert alpha_name not in self.features_in_db, \
            f"{alpha_name} has already been in factor lib! check the name of feature!"
        _, cover_rate = self.load_feature_from_dataframe(feature_data, universe=universe, transformer=transformer,
                                                         cover_rate=True, check_timestamp=True)
        if cover_rate[alpha_name] < COVER_RATE_LIMIT:
            change_reason = f"{alpha_name}'s cover rate is too low and bas been rejected. cover rate:{cover_rate}"
            msg = change_reason
            return 'rejected', change_reason, msg
        self.load_return_data()
        self.feature_name = alpha_name

        status = 'rejected'
        change_reason = ""
        msg = ""
        _periods = EVAL_PERIODS
        ic_flag, trading_direction = self.test_ic(feature_name=alpha_name, output=output)
        if trading_direction is None:
            change_reason = f"{alpha_name} has no clear direction and bas been rejected."
            msg = change_reason
            return 'rejected', change_reason, msg

        if ic_flag:
            corr_flag, names = self.test_correlation(feature_name=alpha_name, method='spearman', verbose=verbose,
                                                     output=output)
            if corr_flag:
                status = 'accepted'
                if names is None:
                    change_reason = f'With no feature in factor lib, ' \
                                    f'{alpha_name} passes all test and has been accepted.'
                else:
                    change_reason = f'{alpha_name} passes all the test and has been accepted.'
                msg = change_reason
            else:
                # compare ic value, correlation test fail
                if names is None:
                    change_reason = f"{alpha_name} has already in factor lib, check your name or rename your alpha!"
                    msg = change_reason
                    return 'rejected', change_reason, msg
                _, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
                alphas_in_db = dict()
                for name in names:
                    ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                    return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                    alphas_in_db[name] = dict()
                    alphas_in_db[name]['ic'] = ic_target
                    alphas_in_db[name]['ret'] = return_target
                status, change_reason, msg, _ = get_feature_status(alpha_name, alphas_in_db, return_info, self.ic_table)

        else:
            return_flag, return_info = self.test_return(alpha_name, trading_direction, verbose=verbose, output=output)
            if return_flag:
                corr_flag, names = self.test_correlation(feature_name=alpha_name, method='spearman', verbose=verbose,
                                                         output=output)
                if corr_flag:
                    status = 'accepted'
                    if names is None:
                        change_reason = f'With no feature in factor lib, ' \
                                        f'{alpha_name} passes all test and has been accepted.'
                    else:
                        change_reason = f'{alpha_name} passes all the test and has been accepted.'
                    msg = change_reason
                else:
                    alphas_in_db = dict()
                    for name in names:
                        ic_target = self.db_api.query_factor_rec_ic(name, period=_periods)
                        return_target = self.db_api.query_factor_rec_return(name, period=_periods)
                        alphas_in_db[name] = dict()
                        alphas_in_db[name]['ic'] = ic_target
                        alphas_in_db[name]['ret'] = return_target
                    status, change_reason, msg, _ = get_feature_status(alpha_name, alphas_in_db, return_info,
                                                                       self.ic_table)

        if self.logger is not None:
            self.logger.info(msg)
        if verbose:
            print(msg)

        return status, change_reason, msg

    @staticmethod
    def _get_date_to_slice(date, num=1):
        year, month = date.year, date.month
        if month + num - 1 > 12:
            end_num = calendar.monthrange(int(year + 1), int(month + num - 13))[1]
            end_date = datetime.datetime(year + 1, month + num - 13, end_num)
        else:
            end_num = calendar.monthrange(int(year), int(month + num - 1))[1]
            end_date = datetime.datetime(year, month + num - 1, end_num)
        if month + num <= 12:
            start_date = datetime.datetime(year, month + num, 1)
        else:
            start_date = datetime.datetime(year + 1, month + num - 12, 1)
        return start_date, end_date

    @staticmethod
    def _calculate_ic(data_set, columns, truncated=None, method='pearson'):
        timestamp, df = data_set
        df = df.reindex(columns=columns)
        target_num = len(columns) - 1
        feature_name = columns[-target_num - 1]
        target_name = columns[-target_num:]

        df_result = df[target_name].corrwith(df[feature_name], method=method) * 100
        df_result = df_result.values.tolist()
        if truncated is not None:
            df.sort_values(by=feature_name, inplace=True, ascending=False)
            df_truncated = df.head(len(df.index) // truncated)
            df_truncated_result = df_truncated[feature_name].corrwith(df_truncated[feature_name], method=method) * 100
            df_result.extend(df_truncated_result.values.tolist())
        df_result.insert(0, timestamp)
        return df_result

    @staticmethod
    def read_files(path_list, pool, scope='Handling data...', handler=pd.read_parquet, time_delta=None,
                   transformer=None, missing_file_allowed=False):

        # Slice stock list
        stock_iter = iter(path_list)
        stock_batch_num = len(path_list)

        # 初始化任务
        result_list = [pool.apply_async(handler, [next(stock_iter), transformer, missing_file_allowed])
                       for _ in range(min(pool._processes, stock_batch_num))]

        df_dict = dict()
        flag = 1
        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        date, alpha_name, df = result_list[out_index].get()
                        if df is not None and len(df) > 0:
                            for timestamp, group in df:
                                if timestamp in df_dict.keys():
                                    df_dict[timestamp].append(group)
                                else:
                                    df_dict[timestamp] = [group]
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description(f"%s | {alpha_name} | {date}" % scope)
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(
                                    pool.apply_async(handler, [next(stock_iter), transformer, missing_file_allowed]))
                            except StopIteration:
                                flag = 0
        # merge
        stock_batch = [(key, df_dict[key], time_delta) for key in df_dict.keys()]
        stock_iter = iter(stock_batch)
        result_list = [pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),))
                       for _ in range(min(pool._processes, len(stock_batch)))]

        df_list = []
        flag = 1
        while len(result_list) > 0:
            time.sleep(0.00001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count = 0
                while index:
                    out_index = index.pop(0) - count
                    df = result_list[out_index].get()
                    df_list.append(df)
                    result_list.pop(out_index)
                    count += 1
                    if flag == 1:
                        try:
                            result_list.append(pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0
        return pd.concat(df_list, copy=False)

    @staticmethod
    def db_reader(path_list, pool, scope='Handling data...', handler=pd.read_parquet, time_delta=None,
                  transformer=None, missing_file_allowed=False):

        # Slice stock list
        stock_iter = iter(path_list)
        stock_batch_num = len(path_list)

        # 初始化任务
        # transformer=None, missing_file_allowed=False, trading_days=None
        result_list = [pool.apply_async(handler, [next(stock_iter), transformer, missing_file_allowed])
                       for _ in range(min(pool._processes, stock_batch_num))]

        df_dict = dict()
        flag = 1
        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        timestamp, df = result_list[out_index].get()
                        df_dict[timestamp] = df

                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description(f"%s | {timestamp}..." % scope)
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(
                                    pool.apply_async(
                                        handler, [next(stock_iter), transformer, missing_file_allowed]))
                            except StopIteration:
                                flag = 0

        # merge
        stock_batch = [(key, df_dict[key], time_delta) for key in df_dict.keys()]
        stock_iter = iter(stock_batch)
        result_list = [pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),))
                       for _ in range(min(pool._processes, len(stock_batch)))]

        df_list = []
        flag = 1
        while len(result_list) > 0:
            time.sleep(0.00001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count = 0
                while index:
                    out_index = index.pop(0) - count
                    df = result_list[out_index].get()
                    df_list.append(df)
                    result_list.pop(out_index)
                    count += 1
                    if flag == 1:
                        try:
                            result_list.append(pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0
        return pd.concat(df_list, copy=False)

    @staticmethod
    def _merge(input_data):
        key, df_list, time_delta = input_data
        df = pd.concat(df_list, copy=False, axis=1, sort=False)
        if time_delta is not None:
            key = time_5m_calculator(key, time_delta)
        df.insert(0, 'timestamp', key)
        df.reset_index(inplace=True)
        df.set_index(['timestamp', 'ticker'], inplace=True)
        return df

    @staticmethod
    def read_files_filter(path_list, universe, pool, scope='Handling data...', handler=pd.read_parquet,
                          time_delta=None, transformer=None,
                          cover_rate=False, missing_file_allowed=False):
        # Slice stock list
        stock_iter = iter(path_list)
        stock_batch_num = len(path_list)

        # 初始化任务
        result_list = [pool.apply_async(
            FeatureAnalysis.load_filter_handler,
            [handler, universe, next(stock_iter), transformer, cover_rate, missing_file_allowed])
            for _ in range(min(pool._processes, stock_batch_num))]

        df_dict = dict()
        cover_rate_dict = dict()
        flag = 1
        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        df, alpha_name, cover_rate_value = result_list[out_index].get()
                        if df is not None and len(df) > 0:
                            for timestamp, group in df:
                                if timestamp in df_dict.keys():
                                    df_dict[timestamp].append(group)
                                else:
                                    df_dict[timestamp] = [group]
                        if cover_rate:
                            if alpha_name in cover_rate_dict.keys():
                                cover_rate_dict[alpha_name].append(cover_rate_value)
                            else:
                                cover_rate_dict[alpha_name] = [cover_rate_value]
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("%s" % scope)
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(FeatureAnalysis.load_filter_handler,
                                                                    [handler, universe, next(stock_iter), transformer,
                                                                     cover_rate, missing_file_allowed]))
                            except StopIteration:
                                flag = 0

        # merge
        stock_batch = [(key, df_dict[key], time_delta) for key in df_dict.keys()]
        stock_iter = iter(stock_batch)
        result_list = [pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),))
                       for _ in range(min(pool._processes, len(stock_batch)))]

        df_list = []
        flag = 1
        while len(result_list) > 0:
            time.sleep(0.00001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count = 0
                while index:
                    out_index = index.pop(0) - count
                    df = result_list[out_index].get()
                    df_list.append(df)
                    result_list.pop(out_index)
                    count += 1
                    if flag == 1:
                        try:
                            result_list.append(pool.apply_async(FeatureAnalysis._merge, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0

        cover_rate_value = dict()
        if cover_rate:
            for key in cover_rate_dict.keys():
                cover_rate_value[key] = np.mean(cover_rate_dict[key])
        else:
            cover_rate_value = None
        return pd.concat(df_list, copy=False), cover_rate_value

    @staticmethod
    def db_reader_filter(path_list, universe, pool, scope='Handling data...', handler=pd.read_parquet,
                         time_delta=None, transformer=None, cover_rate=False, missing_file_allowed=False):
        # Slice stock list
        stock_iter = iter(path_list)
        stock_batch_num = len(path_list)

        # 初始化任务
        result_list = [pool.apply_async(
            FeatureAnalysis.db_filter_handler,
            [handler, universe,
             next(stock_iter), transformer, cover_rate, missing_file_allowed])
            for _ in range(min(pool._processes, stock_batch_num))]

        df_dict = dict()
        cover_rate_dict = dict()
        flag = 1
        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        df, timestamp, cover_rate_value = result_list[out_index].get()
                        df_dict[timestamp] = df
                        if cover_rate:
                            for key in cover_rate_value.keys():
                                if key in cover_rate_dict.keys():
                                    cover_rate_dict[key].append(cover_rate_value[key])
                                else:
                                    cover_rate_dict[key] = [cover_rate_value[key]]

                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("%s" % scope)
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(FeatureAnalysis.db_filter_handler,
                                                                    [handler, universe,
                                                                     next(stock_iter), transformer, cover_rate,
                                                                     missing_file_allowed]))
                            except StopIteration:
                                flag = 0

        df_list = []
        for key in df_dict.keys():
            df = df_dict[key]
            if time_delta is not None:
                key = time_5m_calculator(key, time_delta)
            df.insert(0, 'timestamp', key)
            df_list.append(df)
        df = pd.concat(df_list, copy=False).reset_index().set_index(['timestamp', 'ticker'])

        if cover_rate:
            for key in cover_rate_dict.keys():
                cover_rate_value[key] = np.mean(cover_rate_dict[key])
        else:
            cover_rate_value = None
        return df, cover_rate_value

    @staticmethod
    def load_filter_handler(handler, universe, path, transformer, cover_rate, missing_file_allowed):
        date, alpha_name, df = handler(path, transformer=None, missing_file_allowed=missing_file_allowed, grouped=False)
        if df is None:
            if cover_rate:
                return None, alpha_name, 0
            else:
                return None, alpha_name, None

        stock_list = get_universe(date, universe=universe).index.get_level_values(1)
        df = df[df.index.get_level_values(1).isin(stock_list)]
        if cover_rate:
            cover_rate_value = df.shape[0] / len(stock_list)
        else:
            cover_rate_value = None

        if transformer is not None:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])
            df[alpha_name], _, _ = norm.handle_extreme_value(df[alpha_name].values, method='MAD')
            df[alpha_name] = transformer(df[alpha_name])
        else:
            df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
            df = df.dropna(subset=[alpha_name])

        # daily
        result = [(df.index.get_level_values(0)[0], df.droplevel(0))]

        # min freq
        # grouped_df = df.groupby(level=0, sort=False)
        # result = [(timestamp, group.droplevel(0)) for timestamp, group in grouped_df]
        # print(result)
        return result, alpha_name, cover_rate_value

    @staticmethod
    def load_filter_handler_hft(handler, universe, path, transformer, cover_rate, missing_file_allowed):
        date, alpha_name, df_list = handler(path, transformer=None, missing_file_allowed=missing_file_allowed,
                                            grouped=True)
        if df_list is None or len(df_list) == 0:
            if cover_rate:
                return None, alpha_name, 0
            else:
                return None, alpha_name, None

        cover_rate_value = None
        cover_rate_list = []
        result = []
        for timestamp, df in df_list:
            date = timestamp.date()
            stock_list = get_universe(date, universe=universe).index.get_level_values(1)
            df = df[df.index.isin(stock_list)]
            if cover_rate:
                cover_rate_list.append(df.shape[0] / len(stock_list))

            if transformer is not None:
                df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
                df = df.dropna(subset=[alpha_name])
                df[alpha_name], _, _ = norm.handle_extreme_value(df[alpha_name].values, method='MAD')
                df[alpha_name] = transformer(df[alpha_name])
            else:
                df[alpha_name] = np.where(np.isinf(df[alpha_name]), np.nan, df[alpha_name])
                df = df.dropna(subset=[alpha_name])
            result.append((timestamp, df.droplevel(0)))

        if cover_rate is not None:
            cover_rate_value = np.mean(cover_rate_list)
        return result, alpha_name, cover_rate_value

    @staticmethod
    def db_filter_handler(handler, universe, path, transformer, cover_rate, missing_file_allowed):
        # universe, next(stock_iter), transformer, cover_rate, missing_file_allowed
        timestamp, df = handler(path, transformer=None, missing_file_allowed=missing_file_allowed)
        cover_rate_dict = dict()
        columns = df.columns.tolist()

        def _convert(ser):
            temp = ser.loc[~ser.isna()]
            temp = norm.handle_extreme_value(temp, method='MAD')[0]
            temp = transformer(temp)
            return temp

        cover_rate_value = None

        date = timestamp.date()
        stock_list = get_universe(date, universe=universe).index.get_level_values(1)
        df = df[df.index.isin(stock_list)]

        if transformer is not None:
            df.mask(np.isinf(df), inplace=True)

            for item in columns:
                series = df[item]
                df.loc[~series.isna(), item] = _convert(series)

                if not missing_file_allowed:
                    if df[item].dropna().empty:
                        raise ValueError(f"Data: {item} - {timestamp} is missing!")

                if cover_rate:
                    if item in cover_rate_dict.keys():
                        cover_rate_dict[item].append(series.dropna().shape[0] / len(stock_list))
                    else:
                        cover_rate_dict[item] = [series.dropna().shape[0] / len(stock_list)]

        else:
            df.mask(np.isinf(df), inplace=True)
            for item in columns:
                series = df[item]
                if not missing_file_allowed:
                    if df[item].dropna().empty:
                        raise ValueError(f"Data: {item} - {timestamp} is missing!")

                if cover_rate:
                    if item in cover_rate_dict.keys():
                        cover_rate_dict[item].append(series.dropna().shape[0] / len(stock_list))
                    else:
                        cover_rate_dict[item] = [series.dropna().shape[0] / len(stock_list)]

        if cover_rate is not None:
            cover_rate_value = cover_rate_dict
        return df, timestamp, cover_rate_value

    @staticmethod
    def read_files_target(path_list, pool, scope='Handling data...', handler=pd.read_parquet, transformer=None,
                          filter_trading_status=None):

        # Slice stock list
        stock_iter = iter(path_list)
        stock_batch_num = len(path_list)

        # 初始化任务
        # transformer = None, missing_file_allowed = False, alpha_mode = True, grouped = True
        result_list = [pool.apply_async(
            handler, args=(next(stock_iter), transformer, False, False, False, filter_trading_status))
            for _ in range(min(pool._processes, stock_batch_num))]

        df_list = []
        flag = 1
        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            while len(result_list) > 0:
                time.sleep(0.00001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        _, _, df = result_list[out_index].get()
                        if not df.empty:
                            df_list.append(df)
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("%s" % scope)
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(handler, args=(
                                    next(stock_iter), transformer, False, False, False, filter_trading_status)))
                            except StopIteration:
                                flag = 0
        return pd.concat(df_list, copy=False)

    @staticmethod
    def get_discrete_value(x, fold):
        assert x.ndim == 1, "input array must be 1-d array"
        assert fold >= 2, 'parameter fold must be int and greater than 1!'
        percentile_array = np.arange(0, 100, 100 / fold)[1:]
        value_array = np.percentile(x, percentile_array)

        cond_list = []
        last_flag = None
        for item in value_array:
            if not cond_list:
                cond_list.append(x < item)
                last_flag = item
            else:
                cond_list.append(np.logical_and(x >= last_flag, x <= item))
                last_flag = item
        else:
            cond_list.append(x >= value_array[-1])
        choice_list = list(range(fold))
        return np.select(cond_list, choice_list)

    @staticmethod
    def get_discrete_bi_value(x, fold):
        assert x.ndim == 1, "input array must be 1-d array"
        assert fold >= 2, 'parameter fold must be int and greater than 1!'
        percentile_array = np.arange(0, 100, 100 / fold)[1:]
        value = np.percentile(x, percentile_array)[1]

        cond_list = [x < value, x >= value]
        choice_list = list(range(2))
        return np.select(cond_list, choice_list)

    def reset(self):
        self.feature_name = None
        self.ic_table = None
        self.return_table = None
        self.trading_direction = None
        self.mic_table = None

        self.feature_names = None
        self.feature_data = None
        self.trading_days = None
        self.time_delta = None

        self.target_data = None
        self.data_info = None

        self.ti = self.options.ti


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument('path_way', help='file path of calculating alpha', type=str)
    parse.add_argument('alpha', help='name of alpha', type=str)
    parse.add_argument('start', help='start date for back test', type=str)
    parse.add_argument('end', help='end date for back test', type=str)
    parse.add_argument('path_out', help='path of output folder', type=str)
    parse.add_argument('--ti', help='start trading time by 5 min unit', type=int, default=24)
    parse.add_argument('--tp', help='period of trading by 5 min unit', type=int, default=24)
    parse.add_argument('--negative', help='if the value is true, alpha value will multiply by -1', type=bool,
                       default=False)
    parse.add_argument('--mode', help='mode of back test: long-only, short-only or long-short', type=str,
                       default='long-short')
    parse.add_argument('--end_num', help='end num of stock for back test', type=int, default=200)
    parse.add_argument('--weight', help='weight method for back test', type=str, default='equal',
                       choices=['score', 'equal'])
    parse.add_argument('--universe', help='universe for back test', type=str, default='Float2000',
                       choices=['All', 'Investable', 'HS300', 'ZZ500', 'Float1500', 'Float2000', 'IND', 'MED', 'SER',
                                'COM', 'AGR'])

    args = parse.parse_args()
    FeatureAnalysis.evaluate_feature(path_way=args.path_way
                                     , alpha_name=args.alpha
                                     , bt_start=args.start
                                     , bt_end=args.end
                                     , path_out=args.path_out
                                     , ti=args.ti
                                     , tp=args.tp
                                     , negative=args.negative
                                     , mode=args.mode
                                     , end_num=args.end_num
                                     , weight=args.weight
                                     , universe=args.universe)

    FeatureAnalysis._plot_performance_curve_pic(path_list)
