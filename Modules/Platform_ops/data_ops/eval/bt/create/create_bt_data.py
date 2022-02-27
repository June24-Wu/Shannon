# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
# @author: guichuan
import datetime
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from DataAPI import get_trading_days, get_stock_prices, get_next_trading_day, \
    get_stock_daily_prices, get_index_daily_prices
from TradeTime import TradingTimeStock
from tqdm import tqdm

citics_list = ['CI005001', 'CI005002', 'CI005003', 'CI005004', 'CI005005', 'CI005006', 'CI005007',
               'CI005008', 'CI005009', 'CI005010', 'CI005011', 'CI005012', 'CI005013', 'CI005014',
               'CI005015', 'CI005016', 'CI005017', 'CI005018', 'CI005019', 'CI005020', 'CI005021',
               'CI005022', 'CI005023', 'CI005024', 'CI005025', 'CI005026', 'CI005027', 'CI005028',
               'CI005029', 'CI005030']
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)


def _handle_vwap(input_set):
    df, date, output_path_bt, change_flag = input_set
    if change_flag:
        df_target = df.groupby(level=1).agg({'volume': 'sum', 'amount': 'sum'})
        df_target_new = df.groupby(level=1).trading_status.nth(1)
        df_target = df_target.join(df_target_new, how='right')
    else:
        df_target = df.groupby(level=1).agg({'volume': 'sum', 'amount': 'sum', 'trading_status': 'first'})

    df_target['vwap'] = df_target['amount'] / df_target['volume']
    df_target.reset_index(inplace=True)
    df_target.insert(len(df_target.columns), 'timestamp', int(pd.Timestamp(date).strftime('%Y%m%d')))
    df_target = df_target.set_index(['timestamp', 'ticker'])

    df_pcg = get_stock_daily_prices(date, fields=['pct_chg', 'close', 'fq_factor'], fq=False)
    df_target = df_target.join(df_pcg, how='right')

    df_target['avg'] = df_target['vwap'].fillna(df_target['close'])
    df_target['close'] = df_target['close'] * df_target['fq_factor']
    df_target['trading_status'] = df_target['trading_status'].fillna(3)
    df_target['trading_status'] = df_target['trading_status'].astype(int)
    df_target = df_target.reindex(columns=['pct_chg', 'avg', 'close', 'trading_status', 'fq_factor'])
    df_target = df_target.reset_index().set_index('ticker')

    temp = df_target['avg'] * df_target['fq_factor']
    df_target['avg_2_today_close'] = df_target['close'] / temp
    df_target['last_close_2_avg'] = temp / df_target['close'] * (1 + df_target['pct_chg'] * 0.01)

    df_target = df_target.reset_index().reindex(columns=['timestamp', 'ticker', 'avg', 'trading_status',
                                                         'last_close_2_avg', 'avg_2_today_close'])
    df_target_path2 = df_target.copy()

    for item in ('HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target.insert(len(df_target.columns), 'benchmark_{}_close'.format(item), bench_close_value)
    df_target = df_target.set_index('ticker').sort_index()

    out_path = os.path.join(output_path_bt, 'index', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target.to_parquet(file_name)

    for item in citics_list:
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target_path2.insert(len(df_target_path2.columns), 'benchmark_{}_close'.format(item),
                               bench_close_value)
    out_path = os.path.join(output_path_bt, 'industry', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    df_target_path2 = df_target_path2.set_index('ticker').sort_index()
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target_path2.to_parquet(file_name)

    return date


def _handle_twap(input_set):
    df, date, output_path_bt, change_flag = input_set
    if change_flag:
        df_target = df.groupby(level=1)[['twap']].mean()
        df_target_new = df.groupby(level=1).trading_status.nth(1)
        df_target = df_target.join(df_target_new, how='right')
    else:
        df_target = df.groupby(level=1).agg({'twap': 'mean', 'trading_status': 'first'})

    df_target.reset_index(inplace=True)
    df_target.insert(len(df_target.columns), 'timestamp', int(pd.Timestamp(date).strftime('%Y%m%d')))
    df_target = df_target.set_index(['timestamp', 'ticker'])

    df_pcg = get_stock_daily_prices(date, fields=['pct_chg', 'close', 'fq_factor'], fq=False)
    df_target = df_target.join(df_pcg, how='right')

    df_target['avg'] = df_target['twap'].fillna(df_target['close'])
    df_target['close'] = df_target['close'] * df_target['fq_factor']
    df_target['trading_status'] = df_target['trading_status'].fillna(3)
    df_target['trading_status'] = df_target['trading_status'].astype(int)
    df_target = df_target.reindex(columns=['pct_chg', 'avg', 'close', 'trading_status', 'fq_factor'])
    df_target = df_target.reset_index().set_index('ticker')

    temp = df_target['avg'] * df_target['fq_factor']
    df_target['avg_2_today_close'] = df_target['close'] / temp
    df_target['last_close_2_avg'] = temp / df_target['close'] * (1 + df_target['pct_chg'] * 0.01)

    df_target = df_target.reset_index().reindex(columns=['timestamp', 'ticker', 'avg', 'trading_status',
                                                         'last_close_2_avg', 'avg_2_today_close'])

    df_target_path2 = df_target.copy()

    for item in ('HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target.insert(len(df_target.columns), 'benchmark_{}_close'.format(item), bench_close_value)
    df_target = df_target.set_index('ticker').sort_index()

    out_path = os.path.join(output_path_bt, 'index', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target.to_parquet(file_name)

    for item in citics_list:
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target_path2.insert(len(df_target_path2.columns), 'benchmark_{}_close'.format(item),
                               bench_close_value)
    out_path = os.path.join(output_path_bt, 'industry', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    df_target_path2 = df_target_path2.set_index('ticker').sort_index()
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target_path2.to_parquet(file_name)
    return date


def _handle_close(input_set):
    df, date, output_path_bt, change_flag = input_set
    if change_flag:
        df_target = df.groupby(level=1)[['close']].first()
        df_target_new = df.groupby(level=1).trading_status.nth(1)
        df_target = df_target.join(df_target_new, how='right')
    else:
        df_target = df.groupby(level=1).agg({'close': 'first', 'trading_status': 'first'})

    df_target.reset_index(inplace=True)
    df_target.insert(len(df_target.columns), 'timestamp', int(pd.Timestamp(date).strftime('%Y%m%d')))
    df_target = df_target.set_index(['timestamp', 'ticker'])
    df_target.columns = ['target_avg', 'trading_status']

    df_pcg = get_stock_daily_prices(date, fields=['pct_chg', 'close', 'fq_factor'], fq=False)
    df_target = df_target.join(df_pcg, how='right')

    df_target['avg'] = df_target['target_avg'].fillna(df_target['close'])
    df_target['close'] = df_target['close'] * df_target['fq_factor']
    df_target['trading_status'] = df_target['trading_status'].fillna(3)
    df_target['trading_status'] = df_target['trading_status'].astype(int)
    df_target = df_target.reindex(columns=['pct_chg', 'avg', 'close', 'trading_status', 'fq_factor'])
    df_target = df_target.reset_index().set_index('ticker')

    temp = df_target['avg'] * df_target['fq_factor']
    df_target['avg_2_today_close'] = df_target['close'] / temp
    df_target['last_close_2_avg'] = temp / df_target['close'] * (1 + df_target['pct_chg'] * 0.01)

    df_target = df_target.reset_index().reindex(columns=['timestamp', 'ticker', 'avg', 'trading_status',
                                                         'last_close_2_avg', 'avg_2_today_close'])
    df_target_path2 = df_target.copy()

    for item in ('HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target.insert(len(df_target.columns), 'benchmark_{}_close'.format(item), bench_close_value)
    df_target = df_target.set_index('ticker').sort_index()

    out_path = os.path.join(output_path_bt, 'index', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target.to_parquet(file_name)

    for item in citics_list:
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target_path2.insert(len(df_target_path2.columns), 'benchmark_{}_close'.format(item),
                               bench_close_value)
    out_path = os.path.join(output_path_bt, 'industry', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    df_target_path2 = df_target_path2.set_index('ticker').sort_index()
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target_path2.to_parquet(file_name)
    return date


def _handle_open(input_set):
    df, date, output_path_bt = input_set

    df_target = df.groupby(level=1).agg({'open': 'first', 'trading_status': 'first'})

    df_target.reset_index(inplace=True)
    df_target.insert(len(df_target.columns), 'timestamp', int(pd.Timestamp(date).strftime('%Y%m%d')))
    df_target = df_target.set_index(['timestamp', 'ticker'])
    df_target.columns = ['target_avg', 'trading_status']

    df_pcg = get_stock_daily_prices(date, fields=['pct_chg', 'close', 'fq_factor'], fq=False)
    df_target = df_target.join(df_pcg, how='right')

    df_target['avg'] = df_target['target_avg'].fillna(df_target['close'])
    df_target['close'] = df_target['close'] * df_target['fq_factor']
    df_target['trading_status'] = df_target['trading_status'].fillna(3)
    df_target['trading_status'] = df_target['trading_status'].astype(int)
    df_target = df_target.reindex(columns=['pct_chg', 'avg', 'close', 'trading_status', 'fq_factor'])
    df_target = df_target.reset_index().set_index('ticker')

    temp = df_target['avg'] * df_target['fq_factor']
    df_target['avg_2_today_close'] = df_target['close'] / temp
    df_target['last_close_2_avg'] = temp / df_target['close'] * (1 + df_target['pct_chg'] * 0.01)

    df_target = df_target.reset_index().reindex(columns=['timestamp', 'ticker', 'avg', 'trading_status',
                                                         'last_close_2_avg', 'avg_2_today_close'])
    df_target_path2 = df_target.copy()

    for item in ('HS300', 'ZZ500', 'ZZ800', 'ZZ1000'):
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target.insert(len(df_target.columns), 'benchmark_{}_close'.format(item), bench_close_value)
    df_target = df_target.set_index('ticker').sort_index()

    out_path = os.path.join(output_path_bt, 'index', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target.to_parquet(file_name)

    for item in citics_list:
        bench_close_value = get_index_daily_prices(date, index=item)['close'].iloc[0]
        df_target_path2.insert(len(df_target_path2.columns), 'benchmark_{}_close'.format(item),
                               bench_close_value)
    out_path = os.path.join(output_path_bt, 'industry', str(date.year), '{:0>2}'.format(date.month))
    file_name = os.path.join(out_path, datetime.date.strftime(date, '%Y%m%d') + '.par')
    df_target_path2 = df_target_path2.set_index('ticker').sort_index()
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    df_target_path2.to_parquet(file_name)
    return date


def get_file_name(path_out):
    file_name_list = []
    for root, dirs, files in os.walk(path_out):
        file_name_list.extend(list(map(lambda x: root + '/' + x, files)))
    file_name_list.sort(reverse=False)
    return file_name_list


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def generate_bt_data(prices_info, trading_time, period, time_range_list, gap_end, method_type):
    out_path_folder_bt = os.path.join(out_path_bt, 'ti' + str(trading_time), 'tp' + str(period), method_type)
    if not os.path.exists(out_path_folder_bt): os.makedirs(out_path_folder_bt)

    time_start_hour = time_range_list[0] // 100
    time_start_minute = time_range_list[0] % 100
    time_end_hour = time_range_list[1] // 100
    time_end_minute = time_range_list[1] % 100

    pool = mp.Pool(processes=max_workers)

    if method == 'vwap':
        df_vwap = prices_info.reindex(columns=['volume', 'amount', 'trading_status'])
        df_vwap.sort_index(inplace=True)
        df_start_date = df_vwap.index.get_level_values(0)[0].date()
        df_end_date = df_vwap.index.get_level_values(0)[-1].date()
        df_final_time = pd.Timestamp(df_end_date) + pd.Timedelta(hours=15)

        input_set_list = []
        while True:
            df_start_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                hours=time_start_hour, minutes=time_start_minute)

            if gap_end:
                df_end_time = pd.Timestamp(get_next_trading_day(df_start_date)) + pd.Timedelta(
                    hours=time_end_hour, minutes=time_end_minute)
            else:
                df_end_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                    hours=time_end_hour, minutes=time_end_minute)

            if df_end_time > df_final_time:
                break
            change_flag = 0
            if df_start_time.hour == 9 and df_start_time.minute == 35:
                df_start_time -= pd.Timedelta(minutes=5)
                change_flag = 1
            df_temp = df_vwap.loc[(df_vwap.index.get_level_values(0) < df_end_time) &
                                  (df_vwap.index.get_level_values(0) >= df_start_time)]
            input_set_list.append((df_temp, df_start_date, out_path_folder_bt, change_flag))
            df_start_date = get_next_trading_day(df_start_date)

        stock_iter = iter(input_set_list)
        stock_batch_num = len(input_set_list)

        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            # part 1: delete unused columns
            result_list = [pool.apply_async(_handle_vwap, args=(next(stock_iter),))
                           for _ in range(min(max_workers, stock_batch_num))]

            flag = 1
            while len(result_list) > 0:
                time.sleep(0.0001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        date = result_list[out_index].get()
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("Outputing date: {} for method {}...".format(str(date), method))
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(_handle_vwap, args=(next(stock_iter),)))
                            except StopIteration:
                                flag = 0

    elif method == 'twap':
        df_twap = prices_info.reindex(columns=['twap', 'trading_status'])
        df_twap.sort_index(inplace=True)
        df_start_date = df_twap.index.get_level_values(0)[0].date()
        df_end_date = df_twap.index.get_level_values(0)[-1].date()
        df_final_time = pd.Timestamp(df_end_date) + pd.Timedelta(hours=15)

        input_set_list = []
        while True:
            df_start_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                hours=time_start_hour, minutes=time_start_minute)

            if gap_end:
                df_end_time = pd.Timestamp(get_next_trading_day(df_start_date)) + pd.Timedelta(
                    hours=time_end_hour, minutes=time_end_minute)
            else:
                df_end_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                    hours=time_end_hour, minutes=time_end_minute)

            if df_end_time > df_final_time:
                break
            change_flag = 0
            if df_start_time.hour == 9 and df_start_time.minute == 35:
                df_start_time -= pd.Timedelta(minutes=5)
                change_flag = 1
            df_temp = df_twap.loc[(df_twap.index.get_level_values(0) < df_end_time) &
                                  (df_twap.index.get_level_values(0) >= df_start_time)]
            input_set_list.append((df_temp, df_start_date, out_path_folder_bt, change_flag))
            df_start_date = get_next_trading_day(df_start_date)

        stock_iter = iter(input_set_list)
        stock_batch_num = len(input_set_list)

        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            # part 1: delete unused columns
            result_list = [pool.apply_async(_handle_twap, args=(next(stock_iter),))
                           for _ in range(min(max_workers, stock_batch_num))]

            flag = 1
            while len(result_list) > 0:
                time.sleep(0.0001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        date = result_list[out_index].get()
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("Outputing date: {} for method {}...".format(str(date), method))
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(_handle_twap, args=(next(stock_iter),)))
                            except StopIteration:
                                flag = 0

    elif method == 'close':
        df_close = prices_info.reindex(columns=['close', 'trading_status'])
        df_close.sort_index(inplace=True)
        df_start_date = df_close.index.get_level_values(0)[0].date()
        df_end_date = df_close.index.get_level_values(0)[-1].date()
        df_final_time = pd.Timestamp(df_end_date) + pd.Timedelta(hours=15)

        input_set_list = []
        while True:
            df_start_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                hours=time_start_hour, minutes=time_start_minute)

            if df_start_time > df_final_time:
                break
            change_flag = 0
            if df_start_time.hour == 9 and df_start_time.minute == 30:
                change_flag = 1
                df_temp = df_close.loc[(df_close.index.get_level_values(0) <= df_start_time + pd.Timedelta(minutes=5)) &
                                       (df_close.index.get_level_values(0) >= df_start_time)]
            else:
                df_temp = df_close.loc[df_close.index.get_level_values(0) == df_start_time]

            input_set_list.append((df_temp, df_start_date, out_path_folder_bt, change_flag))
            df_start_date = get_next_trading_day(df_start_date)

        stock_iter = iter(input_set_list)
        stock_batch_num = len(input_set_list)

        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            # part 1: delete unused columns
            result_list = [pool.apply_async(_handle_close, args=(next(stock_iter),))
                           for _ in range(min(max_workers, stock_batch_num))]

            flag = 1
            while len(result_list) > 0:
                time.sleep(0.0001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        date = result_list[out_index].get()
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("Outputing date: {} for method {}...".format(str(date), method))
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(_handle_close, args=(next(stock_iter),)))
                            except StopIteration:
                                flag = 0

    elif method == 'open':
        df_open = prices_info.reindex(columns=['open', 'trading_status'])
        df_open.sort_index(inplace=True)
        df_start_date = df_open.index.get_level_values(0)[0].date()
        df_end_date = df_open.index.get_level_values(0)[-1].date()
        df_final_time = pd.Timestamp(df_end_date) + pd.Timedelta(hours=15)

        input_set_list = []
        while True:
            df_start_time = pd.Timestamp(df_start_date) + pd.Timedelta(
                hours=time_start_hour, minutes=time_start_minute)

            if df_start_time > df_final_time:
                break
            df_temp = df_open.loc[df_open.index.get_level_values(0) == df_start_time]

            input_set_list.append((df_temp, df_start_date, out_path_folder_bt))
            df_start_date = get_next_trading_day(df_start_date)

        stock_iter = iter(input_set_list)
        stock_batch_num = len(input_set_list)

        with tqdm(total=stock_batch_num, ncols=150) as pbar:
            # part 1: delete unused columns
            result_list = [pool.apply_async(_handle_open, args=(next(stock_iter),))
                           for _ in range(min(max_workers, stock_batch_num))]

            flag = 1
            while len(result_list) > 0:
                time.sleep(0.0001)
                status = np.array(list(map(lambda x: x.ready(), result_list)))
                if any(status):
                    index = np.where(status == True)[0].tolist()
                    count = 0
                    while index:
                        out_index = index.pop(0) - count
                        date = result_list[out_index].get()
                        result_list.pop(out_index)
                        count += 1
                        pbar.set_description("Outputing date: {} for method {}...".format(str(date), method))
                        pbar.update(1)
                        if flag == 1:
                            try:
                                result_list.append(pool.apply_async(_handle_open, args=(next(stock_iter),)))
                            except StopIteration:
                                flag = 0

    else:
        raise ValueError(f'Unsupported method: {method}')

    # ?????
    pool.terminate()


if __name__ == '__main__':
    # init
    time_start = time.time()
    out_path_bt = r'/home/DataFolder/Stock/Derivatives/DailyBT'
    methods = ['open']
    max_workers = mp.cpu_count()
    frequency = 500  # frequency is 5min, frequency = 100 is 1min (500 | 00, 100 | 00)
    day_length = 100

    # generate time list
    time_list = []
    TT = TradingTimeStock(freq=frequency)
    while True:
        tt = TT.next_timestamp
        if tt is None:
            break
        else:
            time_list.append(tt // (10 ** 5))
    time_list.append(1500)
    time_list.remove(1300)
    time_list.sort()

    start_date = '2015-01-01'
    end_date = '2021-09-03'
    tis = [24]
    trade_period = [0]
    print(f'Time list has been generated. | Time consume: {round((time.time() - time_start), 4)} secs.')
    time_start = time.time()

    trade_days = get_trading_days(start_date=start_date, end_date=end_date)
    for ti in tis:
        for tp in trade_period:
            assert isinstance(ti, int) and isinstance(tp, int) and 48 >= tp >= 0 and 48 >= ti >= 0, \
                "ti and tp must be integer which ranges from 0 to 48!"
            assert ti + tp <= 48, "trade must be intra-day!"

            bar_num = 240 // (frequency // 100)
            for method in methods:
                if method == 'close':
                    time_list_new = time_list.copy()
                    time_list_new.insert(0, 930)
                    assert tp == 0, f'Trading period for `close` mode must be 0, but {tp} received.'
                    day_gap_end = None
                    time_range = [time_list_new[ti % 49], time_list_new[ti % 49]]

                    num_list = make_batches(len(trade_days), day_length)
                    for start_temp, end_temp in num_list:
                        df_prices = get_stock_prices(start_date=trade_days[start_temp],
                                                     end_date=get_next_trading_day(trade_days[end_temp - 1]),
                                                     fields=['close', 'trading_status'],
                                                     freq='5min',
                                                     fq=False)

                        generate_bt_data(df_prices, ti, tp, time_range, day_gap_end, method)

                        # time
                        time_end = time.time()
                        print(f'Back test data has been generated. ti-{ti}, tp-{tp}, method-{method} | Time consume: '
                              f'{round((time.time() - time_start), 4)} secs.')
                        time_start = time.time()

                else:
                    if method == 'open':
                        assert tp == 0, f'Trading period for `open` mode must be 0, but {tp} received.'
                    day_gap_end = 0
                    if ti + tp >= bar_num:
                        day_gap_end = 1

                    time_range = [time_list[ti % 48], time_list[(ti + tp) % 48]]

                    num_list = make_batches(len(trade_days), day_length)
                    for start_temp, end_temp in num_list:
                        if day_gap_end:
                            end_date_new = get_next_trading_day(trade_days[end_temp - 1])
                        else:
                            end_date_new = trade_days[end_temp - 1]

                        df_prices = get_stock_prices(start_date=trade_days[start_temp],
                                                     end_date=get_next_trading_day(end_date_new),
                                                     fields=['twap', 'volume', 'amount', 'open', 'trading_status'],
                                                     freq='5min',
                                                     fq=False)

                        generate_bt_data(df_prices, ti, tp, time_range, day_gap_end, method)

                        # time
                        time_end = time.time()
                        print(f'Back test data has been generated. ti-{ti}, tp-{tp}, method-{method} | Time consume: '
                              f'{round((time.time() - time_start), 4)} secs.')
                        time_start = time.time()
