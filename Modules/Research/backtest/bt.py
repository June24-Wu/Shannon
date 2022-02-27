# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import datetime
import math
import os
import warnings

import numpy as np
import pandas as pd
from DataAPI import get_trading_days, get_bt_info, convert_to_date, get_universe, get_index_daily_min_bar
from Platform.backtest.metrics.risk_analysis import calc_max_dd, calc_sharpe_ratio
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings('ignore')


class BTDaily:
    def __init__(self, options, start_date=None, end_date=None, logger=None):
        self.options = options
        self.logger = logger
        self.bt_mode = self.options.mode
        self.bt_long_short = self.options.trading_type
        self.score_sorted = self.options.score_sorted
        self.data_format = self.options.data_format

        if not start_date: start_date = self.options.date
        if not end_date: end_date = self.options.end_date
        self.start_date = int(convert_to_date(start_date).strftime('%Y%m%d'))
        self.end_date = int(convert_to_date(end_date).strftime('%Y%m%d'))
        self.bt_days = get_trading_days(start_date=start_date, end_date=end_date, output='int')
        self.method = self.options.bt_price

        self.holding_info = None
        self.cash = self.options.initial_fund
        self.date = None
        self.bt_target_data = None

        self.benchmark_value = dict()
        self.net_value = dict()
        self.alpha_value = dict()
        self.turn_over = dict()
        self.stock_num = dict()
        self.holdings = dict()
        self.missing_day_count = 0

        assert self.options.weight == 'score' or self.options.weight == 'equal', \
            'Unsupported weight method {}: should be equal or score.'.format(str(self.options.weight))

        assert self.bt_long_short in ('long-only', 'short-only', 'long-short'), \
            f'Unsupported long-short mode: {str(self.bt_long_short)}.'

        if self.options.stock_percentage: assert self.options.stock_num <= 1.0, \
            f'Stock_num({self.options.stock_num}) should be less than or equal 1'

        self.missing_allowed = self.options.daily_data_missing_allowed

    def get_bench_cost(self, date):
        if self.options.benchmark in ('SZ50', 'HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
            df = get_index_daily_min_bar(date, self.options.benchmark, freq='5min')
            if self.options.ti == 48:
                avg_price = df['close'].iloc[-1]
            else:
                avg_price = df['open'].iloc[self.options.ti]

        elif self.options.benchmark is "None":
            avg_price = 1

        else:
            # data is missing!
            avg_price = get_bt_info(date, benchmark=self.options.benchmark,
                                    ti=self.options.ti, tp=self.options.trade_period, method=self.method)[
                'benchmark_' + self.options.benchmark + '_close'].iloc[0]
        return avg_price

    def feed_data(self, data):
        assert isinstance(data, pd.DataFrame), 'Wrong data type, input should be DataFrame-type!'
        assert len(data.columns) == 2, 'Wrong data length, input should be columns of 2: ticker and score | order!'
        if self.options.mode == 'intersect':
            data.columns = ['ticker', 'score']
        else:
            data.columns = ['ticker', 'order']
        self.bt_target_data = data.groupby(data.index)

    def run_intersect_long_only(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        if self.data_format == 'dataframe':
            assert self.bt_target_data is not None, 'You should feed score data in dataframe mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    if self.data_format == 'csv':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_csv(file_daily, header=None, names=['ticker', 'score'],
                                         dtype={0: str}).set_index('ticker')['score']
                    elif self.data_format == 'par':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_parquet(file_daily)['score']
                    else:
                        df = self.bt_target_data.get_group(date).set_index('ticker')['score']

                    if math.isclose(df.std(), 0.0):
                        trade_flag = False

                    else:
                        trade_flag = True
                        if not self.score_sorted:
                            df = df.sort_values(ascending=False)

                        if self.options.universe != 'All':
                            df_stock_list = get_universe(date, universe=self.options.universe)
                            stock_list = df_stock_list.index.get_level_values(1)
                            df = df.loc[df.index & stock_list]

                        if self.options.stock_percentage:
                            stock_num = math.floor(df.shape[0] * self.options.stock_num)
                        else:
                            stock_num = self.options.stock_num

                        if self.holding_info is not None:
                            stock_total = stock_num
                            stock_num = math.floor(stock_num * (1 - self.options.keep_pos_percentile))

                        if self.options.keep_pos_percentile != 0.0:
                            df_all = df.copy()

                        if self.options.constant_trading_stock_num:
                            df = df.loc[df.gt(0)]
                        else:
                            df = df.loc[df.gt(0)].nlargest(stock_num, keep='all')

                except FileNotFoundError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'{file_daily} not found!')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        try:
                            df_pcg = df_pcg.loc[df.index]
                        except KeyError:
                            df_pcg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nlargest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array = df.loc[trading_list].values
                                weight = weight_array / np.sum(weight_array)
                            else:
                                dim = len(trading_list)
                                weight = np.full((dim,), 1.0 / dim)

                            market_array = self.options.initial_fund * (1 - self.options.transmission_rate) * \
                                           weight / avg_array // 100.0 * avg_array * 100.0
                            trading_amount = np.sum(market_array)
                            self.cash = self.cash - trading_amount * (1 + self.options.transmission_rate)
                            self.turn_over[date] = trading_amount / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        try:
                            df_trade_cg = df_pcg.loc[df.index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_trade_cg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        status_sr = df_trade_cg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[status_sr[status_sr.eq(0)].index].nlargest(stock_num, keep='all')
                        else:
                            df_target = df.loc[status_sr[status_sr.eq(0)].index]
                        trading_list = df_target.index

                        if self.options.keep_pos_percentile != 0.0:
                            holding_stock = self.holding_info.index
                            share_pos = holding_stock.intersection(trading_list)
                            pos_tradable = df_pcg.loc[df_pcg['trading_status'] == 0].index
                            signal_tradable = df_all.index.intersection(pos_tradable)
                            keep_pos_index = df_all.index.intersection(holding_stock)
                            pos_not_in_target = holding_stock.difference(keep_pos_index).intersection(
                                pos_tradable)
                            remove_pos_index = df_all.loc[holding_stock.intersection(signal_tradable)]. \
                                nsmallest(
                                max(0, stock_num - len(pos_not_in_target) +
                                    len(self.holding_info.index) - stock_total),
                                keep='all').index.union(pos_not_in_target)
                            keep_pos_index = holding_stock.difference(remove_pos_index).difference(share_pos)
                            trading_list = df_all.loc[signal_tradable.difference(holding_stock)].nlargest(
                                stock_num, keep='all').index.union(share_pos)
                        # print(len(keep_pos_index), len(pos_not_in_target), len(remove_pos_index),
                        #       len(trading_list), len(share_pos), stock_num,
                        #       len(self.holding_info.index), )

                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'equal':
                                dim = len(trading_list)
                                df_target = pd.Series(np.full((dim,), 1.0 / dim), index=trading_list)
                            else:
                                df_target = df_target / df_target.sum()

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            if self.options.keep_pos_percentile != 0.0 and len(keep_pos_index) > 0:
                                trading_status_array = np.where(bt_info.index.isin(keep_pos_index), 3,
                                                                trading_status_array)
                            fund_available = np.sum(
                                market_array[np.where(trading_status_array == 0)]) \
                                             * (1 - self.options.transmission_rate - self.options.tax_rate) + self.cash

                            weight = df_target.reindex(index=valid_stock).fillna(0).values
                            target_array = fund_available * (1 - self.options.transmission_rate) * weight
                            avg_today_array = bt_info['avg'].values

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            amount_array = np.where(target_array - market_array > 0,
                                                    ((
                                                             target_array - market_array) / avg_today_array // 100 * avg_today_array
                                                     * 100), target_array - market_array)

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] - self.benchmark_value[date] + 1
                else:
                    relative = (self.net_value[date] - self.net_value[last_day]) / self.net_value[last_day] - \
                               (self.benchmark_value[date] - self.benchmark_value[last_day]) / \
                               self.benchmark_value[last_day]
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def run_intersect_short_only(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        if self.data_format == 'dataframe':
            assert self.bt_target_data is not None, 'You should feed score data in dataframe mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    if self.data_format == 'csv':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_csv(file_daily, header=None, names=['ticker', 'score'],
                                         dtype={0: str}).set_index('ticker')['score']
                    elif self.data_format == 'par':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_parquet(file_daily)['score']
                    else:
                        df = self.bt_target_data.get_group(date).set_index('ticker')['score']

                    trade_flag = True
                    if not self.score_sorted:
                        df = df.sort_values(ascending=False)

                    if self.options.universe != 'All':
                        df_stock_list = get_universe(date, universe=self.options.universe)
                        stock_list = df_stock_list.index.get_level_values(1)
                        df = df.loc[df.index & stock_list]

                    if self.options.stock_percentage:
                        if self.holding_info is not None:
                            stock_num = math.floor(df.shape[0] * self.options.stock_num *
                                                   (1 - self.options.keep_pos_percentile))
                        else:
                            stock_num = math.floor(df.shape[0] * self.options.stock_num)
                    else:
                        if self.holding_info is not None:
                            stock_num = math.floor(self.options.stock_num *
                                                   (1 - self.options.keep_pos_percentile))
                        else:
                            stock_num = self.options.stock_num

                    if self.options.constant_trading_stock_num:
                        df = df.loc[df.lt(0)]
                    else:
                        df = df.loc[df.lt(0)].nsmallest(stock_num, keep='all')

                except FileNotFoundError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'{file_daily} not found!')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        try:
                            df_pcg = df_pcg.loc[df.index]
                        except KeyError:
                            df_pcg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))
                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nsmallest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index

                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1

                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array = df.loc[trading_list].values
                                weight = -weight_array / np.sum(weight_array)
                            else:
                                dim = len(trading_list)
                                weight = np.full((dim,), -1.0 / dim)

                            market_array = \
                                np.ceil(self.options.initial_fund *
                                        (1 - self.options.transmission_rate - self.options.tax_rate) *
                                        weight / avg_array / 100) * avg_array * 100
                            trading_amount = np.sum(market_array)
                            charge_rate = self.options.transmission_rate + self.options.tax_rate
                            self.cash = self.cash - trading_amount * (1 - charge_rate)
                            self.turn_over[date] = np.abs(trading_amount) / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        try:
                            df_trade_cg = df_pcg.loc[df.index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_trade_cg.index).tolist()
                            if self.logger:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        df_target = df_trade_cg['trading_status']
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nsmallest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'equal':
                                dim = len(trading_list)
                                df_target = pd.Series(np.full((dim,), -1.0 / dim), index=df_target.index)
                            else:
                                df_target = -df / df.sum()

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            fund_available = self.cash + np.sum(market_array) + np.sum(market_array[np.where(
                                trading_status_array != 0)]) + np.sum(market_array[np.where(
                                trading_status_array == 0)]) * self.options.transmission_rate
                            fund_available = np.maximum(fund_available, 0)

                            weight = df_target.reindex(index=valid_stock).fillna(0).values
                            target_array = fund_available * (
                                    1 - self.options.transmission_rate - self.options.tax_rate) * weight
                            avg_today_array = bt_info['avg'].values

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            amount_array = np.where(target_array - market_array < 0,
                                                    (np.ceil((
                                                                     target_array - market_array) / avg_today_array / 100) * avg_today_array
                                                     * 100), target_array - market_array)

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] + self.benchmark_value[date] - 1
                else:
                    relative = self.net_value[date] / self.net_value[last_day] \
                               + self.benchmark_value[date] / self.benchmark_value[last_day] - 2
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def run_intersect_long_short(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        if self.data_format == 'dataframe':
            assert self.bt_target_data is not None, 'You should feed score data in dataframe mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    if self.data_format == 'csv':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_csv(file_daily, header=None, names=['ticker', 'score'],
                                         dtype={0: str}).set_index('ticker')['score']
                    elif self.data_format == 'par':
                        file_daily = os.path.join(self.options.score_path, str(date)[0:4],
                                                  ''.join([str(date), '.{}'.format(self.data_format)]))
                        df = pd.read_parquet(file_daily)['score']
                    else:
                        df = self.bt_target_data.get_group(date).set_index('ticker')['score']

                    trade_flag = True
                    if not self.score_sorted:
                        df = df.sort_values(ascending=False)

                    if self.options.universe != 'All':
                        df_stock_list = get_universe(date, universe=self.options.universe)
                        stock_list = df_stock_list.index.get_level_values(1)
                        df = df.loc[df.index & stock_list]

                    if self.options.stock_percentage:
                        stock_num = math.floor((df.shape[0] * self.options.stock_num))
                    else:
                        stock_num = self.options.stock_num

                    if self.options.constant_trading_stock_num:
                        df_buy = df.loc[df.gt(0)]
                        df_sell = df.loc[df.lt(0)]
                    else:
                        df_buy = df.loc[df.gt(0)].nlargest(stock_num, keep='all')
                        df_sell = df.loc[df.lt(0)].nsmallest(stock_num, keep='all')

                except FileNotFoundError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'{file_daily} not found!')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        stock_list_index = df_buy.index | df_sell.index
                        try:
                            df_pcg = df_pcg.loc[stock_list_index]
                        except KeyError:
                            df_pcg = df_pcg.loc[stock_list_index.intersection(df_pcg.index)]
                            stock_missing = stock_list_index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))

                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nlargest(stock_num, keep='all')
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nsmallest(stock_num, keep='all')
                        else:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)]
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)]
                        df_target = pd.concat([df_buy, df_sell])
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1

                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array_buy = df_buy.values
                                weight_buy = weight_array_buy / np.sum(weight_array_buy)
                                weight_array_sell = df_sell.values
                                weight_sell = -weight_array_sell / np.sum(weight_array_sell)
                            else:
                                dim_buy = len(df_buy)
                                weight_buy = np.full((dim_buy,), 1.0 / dim_buy)
                                dim_sell = len(df_sell)
                                weight_sell = np.full((dim_sell,), -1.0 / dim_sell)
                            weight = np.hstack((weight_buy, weight_sell))

                            market_array = np.where(weight > 0,
                                                    np.floor(self.options.initial_fund *
                                                             (1 - self.options.transmission_rate) * weight / avg_array
                                                             / 100) * avg_array * 100,
                                                    np.ceil(self.options.initial_fund *
                                                            (
                                                                    1 - self.options.transmission_rate - self.options.tax_rate) *
                                                            weight / avg_array / 100) * avg_array * 100)
                            trading_amount_long = np.sum(market_array[np.where(market_array > 0)])
                            trading_amount_short = -np.sum(market_array[np.where(market_array < 0)])
                            self.cash += - trading_amount_long / (1 - self.options.transmission_rate) \
                                         + trading_amount_short - trading_amount_short / (
                                                 1 - self.options.transmission_rate - self.options.tax_rate) * \
                                         (self.options.transmission_rate + self.options.tax_rate)
                            self.turn_over[date] = (
                                                           trading_amount_long + trading_amount_short) / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        stock_list_index = df_buy.index | df_sell.index
                        try:
                            df_trade_cg = df_pcg.loc[stock_list_index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[stock_list_index.intersection(df_pcg.index)]
                            stock_missing = stock_list_index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))

                        df_target = df_trade_cg['trading_status']
                        if self.options.constant_trading_stock_num:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nlargest(stock_num, keep='all')
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nsmallest(stock_num, keep='all')
                        else:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)]
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)]
                        df_target = pd.concat([df_buy, df_sell])
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'score':
                                weight_array_buy = df_buy.values
                                weight_buy = weight_array_buy / np.sum(weight_array_buy)
                                weight_array_sell = df_sell.values
                                weight_sell = -weight_array_sell / np.sum(weight_array_sell)
                            else:
                                dim_buy = len(df_buy)
                                if dim_buy == 0:
                                    weight_buy = []
                                else:
                                    weight_buy = np.full((dim_buy,), 1.0 / dim_buy)
                                dim_sell = len(df_sell)
                                if dim_sell == 0:
                                    weight_sell = []
                                else:
                                    weight_sell = np.full((dim_sell,), -1.0 / dim_sell)
                            weight = np.hstack((weight_buy, weight_sell))
                            df_target = pd.Series(weight, index=df_buy.index.append(df_sell.index))

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            weight = df_target.reindex(index=valid_stock).fillna(0).values

                            market_array_buy_available = np.sum(
                                market_array[np.where(np.logical_and(market_array > 0, trading_status_array == 0))])
                            market_array_sell_available = np.sum(
                                market_array[np.where(np.logical_and(market_array < 0, trading_status_array == 0))])

                            fund_available = \
                                self.cash + np.sum(market_array) - \
                                np.abs(np.sum(market_array[np.where(trading_status_array != 0)])) - \
                                market_array_buy_available * (self.options.transmission_rate + self.options.tax_rate) + \
                                market_array_sell_available * self.options.transmission_rate
                            fund_available = np.maximum(fund_available, 0)

                            target_array = np.where(weight > 0,
                                                    fund_available * (1 - self.options.transmission_rate) * weight,
                                                    fund_available * (1 - self.options.transmission_rate
                                                                      - self.options.tax_rate) * weight)

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            # amount_array = np.where(
                            #     target_array - market_array < 0,
                            #     np.ceil((target_array - market_array) / avg_today_array / 100) * avg_today_array * 100,
                            #     np.floor((target_array - market_array) / avg_today_array / 100) * avg_today_array * 100)
                            amount_array = target_array - market_array

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] - self.benchmark_value[date] + 1
                else:
                    relative = self.net_value[date] / self.net_value[last_day] - \
                               self.benchmark_value[date] / self.benchmark_value[last_day]
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def evaluate(self, evalRange=None, verbose=True):
        columns = ['period', 'stock_num', 'return', 'ret_std', 'ret_year', 'sharpe', 'win_ratio', 'max_dd|period',
                   'max_dd_day|date', 'turnover']

        if evalRange is None:
            _stock_num = int(np.mean(list(self.stock_num.values())))
            start_date_format = convert_to_date(self.bt_days[0])
            end_date_format = convert_to_date(self.bt_days[-1])
            days = len(self.bt_days)

            dict_summary = {'date': list(self.alpha_value.keys()), 'alpha_value': list(self.alpha_value.values()),
                            'turnover': list(self.turn_over.values())}
            df = pd.DataFrame(dict_summary).sort_values(by='date')

            base = 1
            _return = (df['alpha_value'].iloc[-1] - base) / base
            _turnover = df['turnover'].mean()
            net_value_list = df['alpha_value'].values / base
            _sharpe = calc_sharpe_ratio(net_value_list.tolist())
            _max_dd, (max_dd_date_start, max_dd_date_end), _max_dd_daily, max_dd_daily_date = \
                calc_max_dd(net_value_list, df['date'].tolist())
            _return_yearly = np.power(np.power(1 + _return, 1 / days), 241) - 1
            _win_ratio_temp = (net_value_list[1:] - net_value_list[0:-1]) / net_value_list[0:-1]
            _return_std = _win_ratio_temp.std() * np.sqrt(252)
            _win_ratio_temp = np.where(_win_ratio_temp >= 0, 1, 0)
            _win_ratio = np.sum(_win_ratio_temp) / len(_win_ratio_temp) * 100

            values = [["{}".format(datetime.datetime.strftime(start_date_format, '%Y%m%d')) + '-' +
                       "{}".format(datetime.datetime.strftime(end_date_format, '%Y%m%d')),
                       _stock_num,
                       _return * 100,
                       _return_std * 100,
                       _return_yearly * 100,
                       _sharpe,
                       _win_ratio,
                       "{:.2f}({:8s}-{:8s})".format(abs(_max_dd) * 100, str(max_dd_date_start), str(max_dd_date_end)),
                       "{:.2f}({:8s})".format(abs(_max_dd_daily) * 100, str(max_dd_daily_date)),
                       _turnover * 100
                       ]]
        else:
            stock_num_df = pd.DataFrame.from_dict(
                {'date': self.stock_num.keys(), 'stock_num': self.stock_num.values()}).set_index('date')[
                'stock_num']

            assert isinstance(evalRange, tuple), 'evalRange must be a tuple, ' \
                                                 'format as ((start1, end1), (start2, end2))'
            dict_summary = {'date': list(self.alpha_value.keys()), 'alpha_value': list(self.alpha_value.values()),
                            'turnover': list(self.turn_over.values())}
            df = pd.DataFrame(dict_summary).sort_values(by='date')

            values = []
            for (start_date, end_date) in evalRange:
                assert isinstance(start_date, int) and start_date >= self.start_date, \
                    f'Query date({start_date}) is earlier than the input data({self.start_date})!'
                assert isinstance(end_date, int) and end_date <= self.end_date, \
                    f'Query date({end_date}) is later than the input data({self.end_date})!'

                start_date_format = convert_to_date(start_date)
                end_date_format = convert_to_date(end_date)

                df_temp = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                base = df_temp['alpha_value'].iloc[0]
                _return = (df_temp['alpha_value'].iloc[-1] - base) / base
                _turnover = df_temp['turnover'].mean()
                net_value_list = df_temp['alpha_value'].values / base
                _sharpe = calc_sharpe_ratio(net_value_list.tolist())
                _max_dd, (max_dd_date_start, max_dd_date_end), _max_dd_daily, max_dd_daily_date = \
                    calc_max_dd(net_value_list, df_temp['date'].tolist())
                days = len(df_temp.index)
                _return_yearly = np.power(np.power(1 + _return, 1 / days), 241) - 1
                _win_ratio_temp = (net_value_list[1:] - net_value_list[0:-1]) / net_value_list[0:-1]
                _return_std = _win_ratio_temp.std() * np.sqrt(252)
                _win_ratio_temp = np.where(_win_ratio_temp >= 0, 1, 0)
                _win_ratio = np.sum(_win_ratio_temp) / len(_win_ratio_temp) * 100

                _stock_num = stock_num_df[
                    (stock_num_df.index >= start_date) & (stock_num_df.index <= end_date)].mean()
                if np.isnan(_stock_num):
                    _stock_num = 0
                else:
                    _stock_num = int(_stock_num)
                values.append(["{}".format(datetime.datetime.strftime(start_date_format, '%Y%m%d')) + '-' +
                               "{}".format(datetime.datetime.strftime(end_date_format, '%Y%m%d')),
                               _stock_num,
                               _return * 100,
                               _return_std * 100,
                               _return_yearly * 100,
                               _sharpe,
                               _win_ratio,
                               "{:.2f}({:8s}-{:8s})".format(abs(_max_dd) * 100, str(max_dd_date_start),
                                                            str(max_dd_date_end)),
                               "{:.2f}({:8s})".format(abs(_max_dd_daily) * 100, str(max_dd_daily_date)),
                               _turnover * 100
                               ])

        df_final = pd.DataFrame(values, columns=columns)
        df_final.set_index('period', inplace=True)

        if verbose:
            print(f'\nBack test summary: "{self.options.trading_type}" mode')
            print(tabulate(df_final, headers=['period'] + df_final.columns.tolist(),
                           tablefmt='grid', floatfmt=".2f", stralign="center", numalign="center"))
        return df, df_final

    def reset(self):
        self.turn_over = dict()
        self.holding_info = None
        self.cash = self.options.initial_fund
        self.date = None
        self.benchmark_value = dict()
        self.net_value = dict()
        self.alpha_value = dict()
        self.stock_num = dict()
        self.holdings = dict()
        self.bt_target_data = None

    def run(self):
        getattr(self, "run_{}_{}".format(self.options.mode, self.options.trading_type.replace('-', '_')))()

    def get_market_value(self, ticker):
        assert isinstance(ticker, str) or isinstance(ticker, list), f"Unsupported ticker type: {type(ticker)}."
        if isinstance(ticker, str): ticker = [ticker]
        if self.holding_info is not None:
            try:
                return self.holding_info.loc[ticker]
            except KeyError:
                return None

    @property
    def market_value(self):
        if self.holding_info is not None:
            return self.holding_info.sum()
        else:
            return None

    @property
    def net_asset(self):
        temp = self.market_value
        if temp is not None:
            return self.cash + temp
        else:
            return None


class BTFeatureDaily:
    def __init__(self, options, start_date=None, end_date=None, logger=None):
        self.options = options
        self.logger = logger
        self.bt_mode = "intersect"
        self.bt_long_short = self.options.trading_type
        self.score_sorted = self.options.score_sorted

        if not start_date: start_date = self.options.date
        if not end_date: end_date = self.options.end_date
        self.start_date = int(convert_to_date(start_date).strftime('%Y%m%d'))
        self.end_date = int(convert_to_date(end_date).strftime('%Y%m%d'))
        self.bt_days = get_trading_days(start_date=start_date, end_date=end_date, output='int')
        self.method = self.options.bt_price

        self.holding_info = None
        self.cash = self.options.initial_fund
        self.date = None
        self.bt_target_data = None

        self.benchmark_value = dict()
        self.net_value = dict()
        self.alpha_value = dict()
        self.turn_over = dict()
        self.stock_num = dict()
        self.holdings = dict()
        self.missing_day_count = 0

        assert self.options.weight == 'score' or self.options.weight == 'equal', \
            'Unsupported weight method {}: should be equal or score.'.format(str(self.options.weight))

        assert self.bt_long_short in ('long-only', 'short-only', 'long-short'), \
            f'Unsupported long-short mode: {str(self.bt_long_short)}.'

        if self.options.stock_percentage: assert self.options.stock_num <= 1.0, \
            f'Stock_num({self.options.stock_num}) should be less than or equal 1'

        self.missing_allowed = self.options.daily_data_missing_allowed

    def get_bench_cost(self, date):
        if self.options.benchmark in ('SZ50', 'HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
            df = get_index_daily_min_bar(date, self.options.benchmark, freq='5min')
            if self.options.ti == 48:
                avg_price = df['close'].iloc[-1]
            else:
                avg_price = df['open'].iloc[self.options.ti]

        elif self.options.benchmark is "None":
            avg_price = 1

        else:
            # data is missing!
            avg_price = get_bt_info(date, benchmark=self.options.benchmark,
                                    ti=self.options.ti, tp=self.options.trade_period, method=self.method)[
                'benchmark_' + self.options.benchmark + '_close'].iloc[0]
        return avg_price

    def feed_data(self, data):
        assert isinstance(data, pd.DataFrame), 'Wrong data type, input should be DataFrame-type!'
        assert len(data.columns) == 2, 'Wrong data length, input should be columns of 2: ticker and score | order!'
        if self.options.mode == 'intersect':
            data.columns = ['ticker', 'score']
        else:
            data.columns = ['ticker', 'order']
        self.bt_target_data = data.groupby(data.index)

    def run_intersect_long_only(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        assert self.bt_target_data is not None, 'You should feed score data in db mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    df = self.bt_target_data.get_group(date).set_index('ticker')['score']
                    if math.isclose(df.std(), 0.0):
                        trade_flag = False
                    else:
                        trade_flag = True
                        if not self.score_sorted:
                            df = df.sort_values(ascending=False)

                        if self.options.universe != 'All':
                            df_stock_list = get_universe(date, universe=self.options.universe)
                            stock_list = df_stock_list.index.get_level_values(1)
                            df = df.loc[df.index & stock_list]

                        if self.options.stock_percentage:
                            stock_num = math.floor((df.shape[0] * self.options.stock_num))
                        else:
                            stock_num = self.options.stock_num

                        if self.options.constant_trading_stock_num:
                            df = df.loc[df.gt(0)]
                        else:
                            df = df.loc[df.gt(0)].nlargest(stock_num, keep='all')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        try:
                            df_pcg = df_pcg.loc[df.index]
                        except KeyError:
                            df_pcg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nlargest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array = df.loc[trading_list].values
                                weight = weight_array / np.sum(weight_array)
                            else:
                                dim = len(trading_list)
                                weight = np.full((dim,), 1.0 / dim)

                            market_array = self.options.initial_fund * (1 - self.options.transmission_rate) * \
                                           weight / avg_array // 100.0 * avg_array * 100.0
                            trading_amount = np.sum(market_array)
                            self.cash = self.cash - trading_amount * (1 + self.options.transmission_rate)
                            self.turn_over[date] = trading_amount / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        try:
                            df_trade_cg = df_pcg.loc[df.index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_trade_cg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        df_target = df_trade_cg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nlargest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'equal':
                                dim = len(trading_list)
                                df_target = pd.Series(np.full((dim,), 1.0 / dim), index=trading_list)
                            else:
                                df_target = df_target / df_target.sum()

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            fund_available = np.sum(
                                market_array[np.where(trading_status_array == 0)]) \
                                             * (1 - self.options.transmission_rate - self.options.tax_rate) + self.cash

                            weight = df_target.reindex(index=valid_stock).fillna(0).values
                            target_array = fund_available * (1 - self.options.transmission_rate) * weight
                            avg_today_array = bt_info['avg'].values

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            amount_array = np.where(target_array - market_array > 0,
                                                    ((
                                                             target_array - market_array) / avg_today_array // 100 * avg_today_array
                                                     * 100), target_array - market_array)

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] - self.benchmark_value[date] + 1
                else:
                    relative = (self.net_value[date] - self.net_value[last_day]) / self.net_value[last_day] - \
                               (self.benchmark_value[date] - self.benchmark_value[last_day]) / \
                               self.benchmark_value[last_day]
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def run_intersect_short_only(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        assert self.bt_target_data is not None, 'You should feed score data in db mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    df = self.bt_target_data.get_group(date).set_index('ticker')['score']

                    trade_flag = True
                    if not self.score_sorted:
                        df = df.sort_values(ascending=False)

                    if self.options.universe != 'All':
                        df_stock_list = get_universe(date, universe=self.options.universe)
                        stock_list = df_stock_list.index.get_level_values(1)
                        df = df.loc[df.index & stock_list]

                    if self.options.stock_percentage:
                        stock_num = math.floor((df.shape[0] * self.options.stock_num))
                    else:
                        stock_num = self.options.stock_num

                    if self.options.constant_trading_stock_num:
                        df = df.loc[df.lt(0)]
                    else:
                        df = df.loc[df.lt(0)].nsmallest(stock_num, keep='all')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        try:
                            df_pcg = df_pcg.loc[df.index]
                        except KeyError:
                            df_pcg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))
                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nsmallest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array = df.loc[trading_list].values
                                weight = -weight_array / np.sum(weight_array)
                            else:
                                dim = len(trading_list)
                                weight = np.full((dim,), -1.0 / dim)

                            market_array = \
                                np.ceil(self.options.initial_fund *
                                        (1 - self.options.transmission_rate - self.options.tax_rate) *
                                        weight / avg_array / 100) * avg_array * 100
                            trading_amount = np.sum(market_array)
                            charge_rate = self.options.transmission_rate + self.options.tax_rate
                            self.cash = self.cash - trading_amount * (1 - charge_rate)
                            self.turn_over[date] = np.abs(trading_amount) / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        try:
                            df_trade_cg = df_pcg.loc[df.index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                            stock_missing = df.index.difference(df_trade_cg.index).tolist()
                            if self.logger:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))

                        df_target = df_trade_cg['trading_status']
                        if self.options.constant_trading_stock_num:
                            df_target = df.loc[df_target[df_target.eq(0)].index].nsmallest(stock_num, keep='all')
                        else:
                            df_target = df.loc[df_target[df_target.eq(0)].index]
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'equal':
                                dim = len(trading_list)
                                df_target = pd.Series(np.full((dim,), -1.0 / dim), index=df_target.index)
                            else:
                                df_target = -df / df.sum()

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: trading delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            fund_available = self.cash + np.sum(market_array) + np.sum(market_array[np.where(
                                trading_status_array != 0)]) + np.sum(market_array[np.where(
                                trading_status_array == 0)]) * self.options.transmission_rate
                            fund_available = np.maximum(fund_available, 0)

                            weight = df_target.reindex(index=valid_stock).fillna(0).values
                            target_array = fund_available * (
                                    1 - self.options.transmission_rate - self.options.tax_rate) * weight
                            avg_today_array = bt_info['avg'].values

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            amount_array = np.where(target_array - market_array < 0,
                                                    (np.ceil((
                                                                     target_array - market_array) / avg_today_array / 100) * avg_today_array
                                                     * 100), target_array - market_array)

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] + self.benchmark_value[date] - 1
                else:
                    relative = self.net_value[date] / self.net_value[last_day] \
                               + self.benchmark_value[date] / self.benchmark_value[last_day] - 2
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def run_intersect_long_short(self):
        last_day = None
        if self.options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + self.options.benchmark + '_close'
            bench_none = False
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]

        assert self.bt_target_data is not None, 'You should feed score data in db mode before your test!'

        with tqdm(total=len(self.bt_days), ncols=150) as pbar:
            for date in self.bt_days:
                try:
                    df = self.bt_target_data.get_group(date).set_index('ticker')['score']

                    trade_flag = True
                    if not self.score_sorted:
                        df = df.sort_values(ascending=False)

                    if self.options.universe != 'All':
                        df_stock_list = get_universe(date, universe=self.options.universe)
                        stock_list = df_stock_list.index.get_level_values(1)
                        df = df.loc[df.index & stock_list]

                    if self.options.stock_percentage:
                        stock_num = math.floor((df.shape[0] * self.options.stock_num))
                    else:
                        stock_num = self.options.stock_num

                    if self.options.constant_trading_stock_num:
                        df_buy = df.loc[df.gt(0)]
                        df_sell = df.loc[df.lt(0)]
                    else:
                        df_buy = df.loc[df.gt(0)].nlargest(stock_num, keep='all')
                        df_sell = df.loc[df.lt(0)].nsmallest(stock_num, keep='all')

                except KeyError:
                    if self.missing_allowed:
                        trade_flag = False
                        self.missing_day_count += 1
                    else:
                        raise FileNotFoundError(f'data of {date} not found!')

                if bench_none:
                    df_pcg = get_bt_info(date, benchmark='ZZ500', ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]
                else:
                    df_pcg = get_bt_info(date, benchmark=self.options.benchmark, ti=self.options.ti,
                                         tp=self.options.trade_period, method=self.method)[columns_bt]

                if self.holding_info is None:
                    if trade_flag:
                        stock_list_index = df_buy.index | df_sell.index
                        try:
                            df_pcg = df_pcg.loc[stock_list_index]
                        except KeyError:
                            df_pcg = df_pcg.loc[stock_list_index.intersection(df_pcg.index)]
                            stock_missing = stock_list_index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))

                        df_target = df_pcg['trading_status'].astype(int)
                        if self.options.constant_trading_stock_num:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nlargest(stock_num, keep='all')
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nsmallest(stock_num, keep='all')
                        else:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)]
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)]
                        df_target = pd.concat([df_buy, df_sell])
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            trade_flag = False
                            self.missing_day_count += 1

                        else:
                            df_pcg = df_pcg.loc[trading_list]
                            avg_array = df_pcg['avg'].values
                            avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                            if self.options.weight == 'score':
                                weight_array_buy = df_buy.values
                                weight_buy = weight_array_buy / np.sum(weight_array_buy)
                                weight_array_sell = df_sell.values
                                weight_sell = -weight_array_sell / np.sum(weight_array_sell)
                            else:
                                dim_buy = len(df_buy)
                                weight_buy = np.full((dim_buy,), 1.0 / dim_buy)
                                dim_sell = len(df_sell)
                                weight_sell = np.full((dim_sell,), -1.0 / dim_sell)
                            weight = np.hstack((weight_buy, weight_sell))

                            market_array = np.where(weight > 0,
                                                    np.floor(self.options.initial_fund *
                                                             (1 - self.options.transmission_rate) * weight / avg_array
                                                             / 100) * avg_array * 100,
                                                    np.ceil(self.options.initial_fund *
                                                            (
                                                                    1 - self.options.transmission_rate - self.options.tax_rate) *
                                                            weight / avg_array / 100) * avg_array * 100)
                            trading_amount_long = np.sum(market_array[np.where(market_array > 0)])
                            trading_amount_short = -np.sum(market_array[np.where(market_array < 0)])
                            self.cash += - trading_amount_long / (1 - self.options.transmission_rate) \
                                         + trading_amount_short - trading_amount_short / (
                                                 1 - self.options.transmission_rate - self.options.tax_rate) * \
                                         (self.options.transmission_rate + self.options.tax_rate)
                            self.turn_over[date] = (
                                                           trading_amount_long + trading_amount_short) / self.options.initial_fund

                            market_array_close = market_array * avg_2_today_close_array
                            self.holding_info = pd.Series(market_array_close, index=df_target.index)

                            bench_initial = 1 / self.get_bench_cost(date)
                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        self.turn_over[date] = 0
                        self.benchmark_value[date] = 1
                        self.net_value[date] = 1
                        self.alpha_value[date] = 1
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                        continue

                else:
                    if trade_flag:
                        stock_list_index = df_buy.index | df_sell.index
                        try:
                            df_trade_cg = df_pcg.loc[stock_list_index]
                        except KeyError:
                            df_trade_cg = df_pcg.loc[stock_list_index.intersection(df_pcg.index)]
                            stock_missing = stock_list_index.difference(df_pcg.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))

                        df_target = df_trade_cg['trading_status']
                        if self.options.constant_trading_stock_num:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nlargest(stock_num, keep='all')
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)] \
                                .nsmallest(stock_num, keep='all')
                        else:
                            df_buy = df_buy.loc[df_buy.index.intersection(df_target[df_target.eq(0)].index)]
                            df_sell = df_sell.loc[df_sell.index.intersection(df_target[df_target.eq(0)].index)]
                        df_target = pd.concat([df_buy, df_sell])
                        trading_list = df_target.index
                        if len(trading_list) == 0:
                            if self.logger is not None:
                                self.logger.warn(f'No valid stock when calculating portfolio return, date: {str(date)}')
                            trade_flag = False
                            self.missing_day_count += 1
                        else:
                            if self.options.weight == 'score':
                                weight_array_buy = df_buy.values
                                weight_buy = weight_array_buy / np.sum(weight_array_buy)
                                weight_array_sell = df_sell.values
                                weight_sell = -weight_array_sell / np.sum(weight_array_sell)
                            else:
                                dim_buy = len(df_buy)
                                if dim_buy == 0:
                                    weight_buy = []
                                else:
                                    weight_buy = np.full((dim_buy,), 1.0 / dim_buy)
                                dim_sell = len(df_sell)
                                if dim_sell == 0:
                                    weight_sell = []
                                else:
                                    weight_sell = np.full((dim_sell,), -1.0 / dim_sell)
                            weight = np.hstack((weight_buy, weight_sell))
                            df_target = pd.Series(weight, index=df_buy.index.append(df_sell.index))

                            valid_stock = self.holding_info.index | trading_list

                            try:
                                bt_info = df_pcg.loc[valid_stock]
                            except KeyError:
                                valid_stock_new = valid_stock & df_pcg.index
                                bt_info = df_pcg.loc[valid_stock_new]
                                stock_missing = valid_stock.difference(bt_info.index).tolist()
                                if self.logger is not None:
                                    for item in stock_missing:
                                        self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                                self.cash += self.holding_info.loc[stock_missing].sum()
                                valid_stock = valid_stock_new

                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)
                            df_holding_array = df_holding.values
                            last_close_2_avg_array = bt_info['last_close_2_avg'].values
                            df_holding_array = df_holding_array * last_close_2_avg_array
                            market_array = df_holding_array
                            trading_status_array = bt_info['trading_status'].values

                            weight = df_target.reindex(index=valid_stock).fillna(0).values

                            market_array_buy_available = np.sum(
                                market_array[np.where(np.logical_and(market_array > 0, trading_status_array == 0))])
                            market_array_sell_available = np.sum(
                                market_array[np.where(np.logical_and(market_array < 0, trading_status_array == 0))])

                            fund_available = \
                                self.cash + np.sum(market_array) - \
                                np.abs(np.sum(market_array[np.where(trading_status_array != 0)])) - \
                                market_array_buy_available * (self.options.transmission_rate + self.options.tax_rate) + \
                                market_array_sell_available * self.options.transmission_rate
                            fund_available = np.maximum(fund_available, 0)

                            target_array = np.where(weight > 0,
                                                    fund_available * (1 - self.options.transmission_rate) * weight,
                                                    fund_available * (1 - self.options.transmission_rate
                                                                      - self.options.tax_rate) * weight)

                            # untradable
                            target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                    market_array, target_array)

                            # amount_array = np.where(
                            #     target_array - market_array < 0,
                            #     np.ceil((target_array - market_array) / avg_today_array / 100) * avg_today_array * 100,
                            #     np.floor((target_array - market_array) / avg_today_array / 100) * avg_today_array * 100)
                            amount_array = target_array - market_array

                            if self.options.change_pos_threshold != 0:
                                amount_array = np.where(
                                    abs(amount_array / market_array) < self.options.change_pos_threshold,
                                    0, amount_array)
                            sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                            buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                            self.cash += sell_amount * (1 - self.options.transmission_rate - self.options.tax_rate) - \
                                         buy_amount / (1 - self.options.transmission_rate)
                            market_array = market_array + amount_array
                            self.turn_over[date] = (sell_amount + buy_amount) / self.net_asset / 2

                            avg_2_today_close_array = bt_info['avg_2_today_close'].values
                            market_array_close = market_array * avg_2_today_close_array
                            df_holding = pd.Series(market_array_close, index=valid_stock)
                            self.holding_info = df_holding[~df_holding.eq(0)]

                            if bench_none:
                                bench_close = 1
                            else:
                                bench_close = df_pcg[bench_name].iloc[0]
                            self.benchmark_value[date] = bench_close * bench_initial

                    if not trade_flag:
                        valid_stock = self.holding_info.index
                        try:
                            bt_info = df_pcg.loc[valid_stock]
                            df_holding = self.holding_info
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            if self.logger is not None:
                                for item in stock_missing:
                                    self.logger.warn('Date {}: holding delisted stock: {}'.format(str(date), item))
                            self.cash += self.holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new
                            df_holding = self.holding_info.reindex(index=valid_stock).fillna(0)

                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                        self.turn_over[date] = 0
                        self.holding_info = pd.Series(market_array, index=valid_stock)

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        self.benchmark_value[date] = bench_close * bench_initial

                self.net_value[date] = self.net_asset / self.options.initial_fund
                if last_day is None:
                    self.alpha_value[date] = self.net_value[date] - self.benchmark_value[date] + 1
                else:
                    relative = self.net_value[date] / self.net_value[last_day] - \
                               self.benchmark_value[date] / self.benchmark_value[last_day]
                    self.alpha_value[date] = self.alpha_value[last_day] * (1 + relative)
                self.holdings[date] = self.holding_info.copy()
                self.stock_num[date] = self.holding_info.shape[0]
                last_day = date

                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

    def evaluate(self, evalRange=None, verbose=True):
        columns = ['period', 'stock_num', 'return', 'ret_std', 'ret_year', 'sharpe', 'win_ratio', 'max_dd|period',
                   'max_dd_day|date', 'turnover']

        if evalRange is None:
            _stock_num = int(np.mean(list(self.stock_num.values())))
            start_date_format = convert_to_date(self.bt_days[0])
            end_date_format = convert_to_date(self.bt_days[-1])
            days = len(self.bt_days)

            dict_summary = {'date': list(self.alpha_value.keys()), 'alpha_value': list(self.alpha_value.values()),
                            'turnover': list(self.turn_over.values())}
            df = pd.DataFrame(dict_summary).sort_values(by='date')

            base = 1
            _return = (df['alpha_value'].iloc[-1] - base) / base
            _turnover = df['turnover'].mean()
            net_value_list = df['alpha_value'].values / base
            _sharpe = calc_sharpe_ratio(net_value_list.tolist())
            _max_dd, (max_dd_date_start, max_dd_date_end), _max_dd_daily, max_dd_daily_date = \
                calc_max_dd(net_value_list, df['date'].tolist())
            _return_yearly = np.power(np.power(1 + _return, 1 / days), 241) - 1
            _win_ratio_temp = (net_value_list[1:] - net_value_list[0:-1]) / net_value_list[0:-1]
            _return_std = _win_ratio_temp.std() * np.sqrt(252)
            _win_ratio_temp = np.where(_win_ratio_temp >= 0, 1, 0)
            _win_ratio = np.sum(_win_ratio_temp) / len(_win_ratio_temp) * 100

            values = [["{}".format(datetime.datetime.strftime(start_date_format, '%Y%m%d')) + '-' +
                       "{}".format(datetime.datetime.strftime(end_date_format, '%Y%m%d')),
                       _stock_num,
                       _return * 100,
                       _return_std * 100,
                       _return_yearly * 100,
                       _sharpe,
                       _win_ratio,
                       "{:.2f}({:8s}-{:8s})".format(abs(_max_dd) * 100, str(max_dd_date_start), str(max_dd_date_end)),
                       "{:.2f}({:8s})".format(abs(_max_dd_daily) * 100, str(max_dd_daily_date)),
                       _turnover * 100
                       ]]
        else:

            stock_num_df = pd.DataFrame.from_dict(
                {'date': self.stock_num.keys(), 'stock_num': self.stock_num.values()}).set_index('date')[
                'stock_num']

            assert isinstance(evalRange, tuple), 'evalRange must be a tuple, ' \
                                                 'format as ((start1, end1), (start2, end2))'
            dict_summary = {'date': list(self.alpha_value.keys()), 'alpha_value': list(self.alpha_value.values()),
                            'turnover': list(self.turn_over.values())}
            df = pd.DataFrame(dict_summary).sort_values(by='date')

            values = []
            for (start_date, end_date) in evalRange:
                assert isinstance(start_date, int) and start_date >= self.start_date, \
                    f'Query date({start_date}) is earlier than the input data({self.start_date})!'
                assert isinstance(end_date, int) and end_date <= self.end_date, \
                    f'Query date({end_date}) is later than the input data({self.end_date})!'

                start_date_format = convert_to_date(start_date)
                end_date_format = convert_to_date(end_date)

                df_temp = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                base = df_temp['alpha_value'].iloc[0]
                _return = (df_temp['alpha_value'].iloc[-1] - base) / base
                _turnover = df_temp['turnover'].mean()
                net_value_list = df_temp['alpha_value'].values / base
                _sharpe = calc_sharpe_ratio(net_value_list.tolist())
                _max_dd, (max_dd_date_start, max_dd_date_end), _max_dd_daily, max_dd_daily_date = \
                    calc_max_dd(net_value_list, df_temp['date'].tolist())
                days = len(df_temp.index)
                _return_yearly = np.power(np.power(1 + _return, 1 / days), 241) - 1
                _win_ratio_temp = (net_value_list[1:] - net_value_list[0:-1]) / net_value_list[0:-1]
                _return_std = _win_ratio_temp.std() * np.sqrt(252)
                _win_ratio_temp = np.where(_win_ratio_temp >= 0, 1, 0)
                _win_ratio = np.sum(_win_ratio_temp) / len(_win_ratio_temp) * 100

                _stock_num = stock_num_df[
                    (stock_num_df.index >= start_date) & (stock_num_df.index <= end_date)].mean()
                if np.isnan(_stock_num):
                    _stock_num = 0
                else:
                    _stock_num = int(_stock_num)
                values.append(["{}".format(datetime.datetime.strftime(start_date_format, '%Y%m%d')) + '-' +
                               "{}".format(datetime.datetime.strftime(end_date_format, '%Y%m%d')),
                               _stock_num,
                               _return * 100,
                               _return_std * 100,
                               _return_yearly * 100,
                               _sharpe,
                               _win_ratio,
                               "{:.2f}({:8s}-{:8s})".format(abs(_max_dd) * 100, str(max_dd_date_start),
                                                            str(max_dd_date_end)),
                               "{:.2f}({:8s})".format(abs(_max_dd_daily) * 100, str(max_dd_daily_date)),
                               _turnover * 100
                               ])

        df_final = pd.DataFrame(values, columns=columns)
        df_final.set_index('period', inplace=True)

        if verbose:
            print(f'\nBack test summary: "{self.options.trading_type}" mode')
            print(tabulate(df_final, headers=['period'] + df_final.columns.tolist(),
                           tablefmt='grid', floatfmt=".2f", stralign="center", numalign="center"))
        return df, df_final

    def reset(self):
        self.turn_over = dict()
        self.holding_info = None
        self.cash = self.options.initial_fund
        self.date = None
        self.benchmark_value = dict()
        self.net_value = dict()
        self.alpha_value = dict()
        self.stock_num = dict()
        self.holdings = dict()
        self.bt_target_data = None

    def run(self):
        getattr(self, "run_{}_{}".format(self.options.mode, self.options.trading_type.replace('-', '_')))()

    def run_group_test(self, group=5, tax_rate=0.0, transmission_rate=0.0):
        assert isinstance(group, int) and group >= 2, f"group num should be int larger than 1, but {group} received!"
        import multiprocessing as mp
        import time
        pool = mp.Pool(processes=group)

        assert self.bt_target_data is not None, 'You should feed score data before your test!'
        base_dis = 1 / group
        group_quantile = np.arange(0, 1 + base_dis, base_dis)
        group_quantile = [group_quantile[i:i + 2] for i in range(len(group_quantile) - 1)]
        configs = self.options
        configs.tax_rate = tax_rate
        configs.transmission_rate = transmission_rate

        # ?????
        # bt_days, bt_target_data, group_quantile, options, logger=None, bar=False
        bars = [True if i == 0 else False for i in range(group)]
        result_list = [pool.apply_async(BTFeatureDaily.group_handler,
                                        args=(self.bt_days, self.bt_target_data, group_quantile[i], i, configs,
                                              bars[i],))
                       for i in range(group)]

        final_result = []
        msg_list = []
        holdings_info = dict()
        while len(result_list) > 0:
            time.sleep(0.00001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count = 0
                while index:
                    out_index = index.pop(0) - count
                    result, holdings, i, warm_msgs = result_list[out_index].get()
                    final_result.append(result)
                    if warm_msgs:
                        msg_list.extend(warm_msgs)
                    holdings_info[i] = holdings
                    result_list.pop(out_index)
                    count += 1

        df = final_result[0]
        for i in final_result[1:]:
            df = df.join(i, how='outer')
        columns = []
        for item in ("alpha_group", "net_group", "turnover_group", "stock_num_group"):
            for i in range(group):
                columns.append(item + str(i))
        columns.append("benchmark_value")
        if self.logger is not None:
            for item in msg_list:
                self.logger.warn(item)
        pool.close()
        pool.join()
        return df.reindex(columns=columns), holdings_info

    def get_market_value(self, ticker):
        assert isinstance(ticker, str) or isinstance(ticker, list), f"Unsupported ticker type: {type(ticker)}."
        if isinstance(ticker, str): ticker = [ticker]
        if self.holding_info is not None:
            try:
                return self.holding_info.loc[ticker]
            except KeyError:
                return None

    @property
    def market_value(self):
        if self.holding_info is not None:
            return self.holding_info.sum()
        else:
            return None

    @property
    def net_asset(self):
        temp = self.market_value
        if temp is not None:
            return self.cash + temp
        else:
            return None

    @staticmethod
    def group_handler(bt_days, bt_target_data, group_quantile, group_num, options, bar=False):
        last_day = None
        if options.benchmark == 'None':
            bench_name = 'benchmark_ZZ500_close'
            bench_none = True
        else:
            bench_name = 'benchmark_' + options.benchmark + '_close'
            bench_none = False
        method = options.bt_price
        score_sorted = options.score_sorted
        columns_bt = ['avg', 'trading_status', 'last_close_2_avg', 'avg_2_today_close', bench_name]
        if bar:
            pbar = tqdm(total=len(bt_days), ncols=150)

        # inital data
        holding_info = None
        turn_over = dict()
        cash = options.initial_fund
        benchmark_value = dict()
        net_value = dict()
        alpha_value = dict()
        stock_num = dict()
        holdings = dict()
        warn_msg = []

        def get_bench_cost(query_date, option_bt):
            if option_bt.benchmark in ('SZ50', 'HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
                df = get_index_daily_min_bar(query_date, option_bt.benchmark, freq='5min')
                if option_bt.ti == 48:
                    avg_price = df['close'].iloc[-1]
                else:
                    avg_price = df['open'].iloc[option_bt.ti]

            elif option_bt.benchmark is "None":
                avg_price = 1

            else:
                # data is missing!
                avg_price = get_bt_info(query_date, benchmark=option_bt.benchmark,
                                        ti=option_bt.ti, tp=option_bt.trade_period, method=method)[
                    'benchmark_' + option_bt.benchmark + '_close'].iloc[0]
            return avg_price

        def market_value(holding_info_in):
            if holding_info_in is not None:
                return holding_info_in.sum()
            else:
                return None

        def get_net_asset(holding_info_in, cash_in):
            temp = market_value(holding_info_in)
            if temp is not None:
                return cash_in + temp
            else:
                return None

        for date in bt_days:
            try:
                df = bt_target_data.get_group(date).set_index('ticker')['score']

                if math.isclose(df.std(), 0.0):
                    trade_flag = False

                else:
                    trade_flag = True
                    if not score_sorted:
                        df = df.sort_values(ascending=False)
    
                    if options.universe != 'All':
                        df_stock_list = get_universe(date, universe=options.universe)
                        stock_list = df_stock_list.index.get_level_values(1)
                        df = df.loc[df.index & stock_list]

                    score_quantile = df.quantile(group_quantile)
                    item1, item2 = score_quantile.values.tolist()
                    if item2 == 1.0:
                        df = df.loc[(df <= item2) & (df >= item1)]
                    elif item2 == item1:
                        df = df.loc[df == item1]
                    else:
                        df = df.loc[(df < item2) & (df >= item1)]

            except KeyError:
                trade_flag = False

            if bench_none:
                df_pcg = get_bt_info(date, benchmark='ZZ500', ti=options.ti,
                                     tp=options.trade_period, method=method)[columns_bt]
            else:
                df_pcg = get_bt_info(date, benchmark=options.benchmark, ti=options.ti,
                                     tp=options.trade_period, method=method)[columns_bt]

            if holding_info is None:
                if trade_flag:
                    try:
                        df_pcg = df_pcg.loc[df.index]
                    except KeyError:
                        df_pcg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                        stock_missing = df.index.difference(df_pcg.index).tolist()
                        for item in stock_missing:
                            warn_msg.append('Date {}: trading delisted stock: {}'.format(str(date), item))

                    df_target = df_pcg['trading_status'].astype(int)
                    df_target = df.loc[df_target[df_target.eq(0)].index]
                    trading_list = df_target.index
                    if len(trading_list) == 0:
                        warn_msg.append(f'No valid stock when calculating portfolio return, date: {str(date)}')
                        trade_flag = False
                    else:
                        df_pcg = df_pcg.loc[trading_list]
                        avg_array = df_pcg['avg'].values
                        avg_2_today_close_array = df_pcg['avg_2_today_close'].values

                        dim = len(trading_list)
                        weight = np.full((dim,), 1.0 / dim)

                        market_array = options.initial_fund * (1 - options.transmission_rate) * \
                                       weight / avg_array // 100.0 * avg_array * 100.0
                        trading_amount = np.sum(market_array)
                        cash = cash - trading_amount * (1 + options.transmission_rate)
                        turn_over[date] = trading_amount / options.initial_fund

                        market_array_close = market_array * avg_2_today_close_array
                        holding_info = pd.Series(market_array_close, index=df_target.index)

                        bench_initial = 1 / get_bench_cost(date, options)
                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        benchmark_value[date] = bench_close * bench_initial

                if not trade_flag:
                    turn_over[date] = 0
                    benchmark_value[date] = 1
                    net_value[date] = 1
                    alpha_value[date] = 1
                    if bar:
                        pbar.set_description("Getting BT result for date: {}...".format(date))
                        pbar.update(1)
                    continue

            else:
                if trade_flag:
                    try:
                        df_trade_cg = df_pcg.loc[df.index]
                    except KeyError:
                        df_trade_cg = df_pcg.loc[df.index.intersection(df_pcg.index)]
                        stock_missing = df.index.difference(df_trade_cg.index).tolist()
                        for item in stock_missing:
                            warn_msg.append('Date {}: trading delisted stock: {}'.format(str(date), item))

                    df_target = df_trade_cg['trading_status'].astype(int)
                    df_target = df.loc[df_target[df_target.eq(0)].index]
                    trading_list = df_target.index
                    if len(trading_list) == 0:
                        trade_flag = False
                    else:
                        dim = len(trading_list)
                        df_target = pd.Series(np.full((dim,), 1.0 / dim), index=trading_list)

                        valid_stock = holding_info.index | trading_list

                        try:
                            bt_info = df_pcg.loc[valid_stock]
                        except KeyError:
                            valid_stock_new = valid_stock & df_pcg.index
                            bt_info = df_pcg.loc[valid_stock_new]
                            stock_missing = valid_stock.difference(bt_info.index).tolist()
                            for item in stock_missing:
                                warn_msg.append('Date {}: trading delisted stock: {}'.format(str(date), item))
                            cash += holding_info.loc[stock_missing].sum()
                            valid_stock = valid_stock_new

                        df_holding = holding_info.reindex(index=valid_stock).fillna(0)
                        df_holding_array = df_holding.values
                        last_close_2_avg_array = bt_info['last_close_2_avg'].values
                        df_holding_array = df_holding_array * last_close_2_avg_array
                        market_array = df_holding_array
                        trading_status_array = bt_info['trading_status'].values

                        fund_available = np.sum(
                            market_array[np.where(trading_status_array == 0)]) \
                                         * (1 - options.transmission_rate - options.tax_rate) + cash

                        weight = df_target.reindex(index=valid_stock).fillna(0).values
                        target_array = fund_available * (1 - options.transmission_rate) * weight
                        avg_today_array = bt_info['avg'].values

                        # untradable
                        target_array = np.where(np.logical_and(market_array != 0, trading_status_array != 0),
                                                market_array, target_array)

                        amount_array = np.where(target_array - market_array > 0,
                                                ((
                                                         target_array - market_array) / avg_today_array // 100 * avg_today_array
                                                 * 100), target_array - market_array)

                        if options.change_pos_threshold != 0:
                            amount_array = np.where(
                                abs(amount_array / market_array) < options.change_pos_threshold,
                                0, amount_array)
                        sell_amount = -np.sum(amount_array[np.where(amount_array < 0)])
                        buy_amount = np.sum(amount_array[np.where(amount_array > 0)])
                        cash += sell_amount * (1 - options.transmission_rate - options.tax_rate) - \
                                buy_amount / (1 - options.transmission_rate)
                        market_array = market_array + amount_array
                        net_asset = get_net_asset(holding_info, cash)
                        turn_over[date] = (sell_amount + buy_amount) / net_asset / 2

                        avg_2_today_close_array = bt_info['avg_2_today_close'].values
                        market_array_close = market_array * avg_2_today_close_array
                        df_holding = pd.Series(market_array_close, index=valid_stock)
                        holding_info = df_holding[~df_holding.eq(0)]

                        if bench_none:
                            bench_close = 1
                        else:
                            bench_close = df_pcg[bench_name].iloc[0]
                        benchmark_value[date] = bench_close * bench_initial

                if not trade_flag:
                    valid_stock = holding_info.index
                    try:
                        bt_info = df_pcg.loc[valid_stock]
                        df_holding = holding_info
                    except KeyError:
                        valid_stock_new = valid_stock & df_pcg.index
                        bt_info = df_pcg.loc[valid_stock_new]
                        stock_missing = valid_stock.difference(bt_info.index).tolist()
                        for item in stock_missing:
                            warn_msg.append('Date {}: trading delisted stock: {}'.format(str(date), item))
                        cash += holding_info.loc[stock_missing].sum()
                        valid_stock = valid_stock_new
                        df_holding = holding_info.reindex(index=valid_stock).fillna(0)

                    df_holding_array = df_holding.values
                    last_close_2_avg_array = bt_info['last_close_2_avg'].values
                    avg_2_today_close_array = bt_info['avg_2_today_close'].values
                    market_array = df_holding_array * last_close_2_avg_array * avg_2_today_close_array
                    turn_over[date] = 0
                    holding_info = pd.Series(market_array, index=valid_stock)

                    if bench_none:
                        bench_close = 1
                    else:
                        bench_close = df_pcg[bench_name].iloc[0]
                    benchmark_value[date] = bench_close * bench_initial

            net_asset = get_net_asset(holding_info, cash)
            net_value[date] = net_asset / options.initial_fund
            if last_day is None:
                alpha_value[date] = net_value[date] - benchmark_value[date] + 1
            else:
                relative = (net_value[date] - net_value[last_day]) / net_value[last_day] - \
                           (benchmark_value[date] - benchmark_value[last_day]) / \
                           benchmark_value[last_day]
                alpha_value[date] = alpha_value[last_day] * (1 + relative)
            holdings[date] = holding_info.copy()
            stock_num[date] = holding_info.shape[0]
            last_day = date

            if bar:
                pbar.set_description("Getting BT result for date: {}...".format(date))
                pbar.update(1)

        if group_num == 0:
            df = pd.DataFrame([alpha_value, net_value, benchmark_value, turn_over, stock_num]).T.reset_index()
            df.columns = ['timestamp', f'alpha_group{group_num}', f'net_group{group_num}',
                          f'benchmark_value', f'turnover_group{group_num}', f'stock_num_group{group_num}']
        else:
            df = pd.DataFrame([alpha_value, net_value, turn_over, stock_num]).T.reset_index()
            df.columns = ['timestamp', f'alpha_group{group_num}', f'net_group{group_num}',
                          f'turnover_group{group_num}', f'stock_num_group{group_num}']
        df.dropna(subset=[f'stock_num_group{group_num}'], inplace=True)
        df[f'stock_num_group{group_num}'] = df[f'stock_num_group{group_num}'].astype(int)
        df.set_index('timestamp', inplace=True)
        return df, holdings, group_num, warn_msg
