# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
# version: 2022.02.10
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from DevTools.tools.elk_logger import get_elk_logger
from ExObject.DateTime import DateTime
from ExObject.TimeSpan import TimeSpan
from Research.feature.ft import FeatureAnalysis
from tqdm import tqdm

from correlation.config.config import configs_corr


class CorrService:
    def __init__(self, options, logger=None):
        self.configs = options

        # logger
        if logger is None:
            self.logger = get_elk_logger("platform_correlation_service_test", console=True)
        else:
            self.logger = logger

        # corr info
        self.corr_matrix_dict = dict()
        self.corr_end_time_dict = dict()
        self.corr_start_time_dict = dict()
        self.last_save = DateTime.Now()

        # corr path
        self.corr_path_dict = {
            category: os.path.join(self.configs.corr_path, self.configs.db, "corr", category, "corr.par")
            for category in self.configs.category}

        # load corr
        self.load_corr_matrix()

    def load_corr_matrix(self):
        for key in self.corr_path_dict.keys():
            corr_path = self.corr_path_dict[key]
            try:
                corr_matrix = pd.read_parquet(corr_path)
                self.corr_matrix_dict[key] = corr_matrix
                self.corr_end_time_dict[key] = corr_matrix.index.get_level_values(0).max()
                self.corr_start_time_dict[key] = corr_matrix.index.get_level_values(0).min()
            except FileNotFoundError:
                if not os.path.exists(os.path.dirname(corr_path)):
                    os.makedirs(os.path.dirname(corr_path))

    def calculate_correlation(self, feature_name, df, category):
        self.logger.info("=" * 125)
        self.logger.info(f"Calculate correlation of {feature_name}, length: {df.shape[0]}...")
        self.logger.info("=" * 125)

        result = CorrService._get_feature_correlation(df, self.corr_matrix_dict[category])

        self.logger.info(
            f'Correlation result of {feature_name} has been finished.')
        return result

    def save_corr_matrix(self):
        if (DateTime.Now() - self.last_save) > TimeSpan(minute=self.configs.SAVE_TIME_SPAN_MINUTES):
            for key in self.corr_matrix_dict:
                corr_matrix = self.corr_matrix_dict[key]
                path_output = self.corr_path_dict[key]
                corr_matrix.to_parquet(path_output)
            self.last_save = DateTime.Now()
            self.logger.info('Corr_matrix has been updated.')

    @staticmethod
    def _get_feature_correlation(feature_data, corr_matrix):
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

    def append_corr_matrix(self, alpha_name, df, category):
        if category not in self.corr_matrix_dict.keys():
            end_time = df.index.get_level_values(0).max()
            self.corr_end_time_dict[category] = end_time

            start_time = pd.Timestamp(end_time.year, end_time.month, end_time.day) - \
                         pd.Timedelta(days=getattr(self.configs, category)["CORR_PERIOD"])
            self.corr_start_time_dict[category] = start_time
            self.corr_matrix_dict[category] = df.loc[
                (df.index.get_level_values(0) >= start_time) &
                (df.index.get_level_values(0) <= end_time)].sort_index()
        else:
            start_time = self.corr_start_time_dict[category]
            end_time = self.corr_end_time_dict[category]
            self.corr_matrix_dict[category] = self.corr_matrix_dict[category].join(
                df.loc[(df.index.get_level_values(0) >= start_time) &
                       (df.index.get_level_values(0) <= end_time)], how='outer')

        self.logger.info(
            f'Operation for corr_matrix of {alpha_name} has done. status: accepted_no_replace.')
        self.save_corr_matrix()

    def replace_corr_matrix(self, alpha_name, df, category):
        start_time = self.corr_start_time_dict[category]
        end_time = self.corr_end_time_dict[category]
        corr_matrix = self.corr_matrix_dict[category].drop(columns=alpha_name)
        self.corr_matrix_dict[category] = corr_matrix.join(
            df.loc[(df.index.get_level_values(0) <= end_time) &
                   (df.index.get_level_values(0) >= start_time)], how='outer')

        self.logger.info(
            f'Operation for corr_matrix of {alpha_name} has done. status: accepted_with_replace.')
        self.save_corr_matrix()


if __name__ == '__main__':
    configs_test = configs_corr
    service = CorrService(configs_test)
    print(service.corr_matrix_dict["PV"])
