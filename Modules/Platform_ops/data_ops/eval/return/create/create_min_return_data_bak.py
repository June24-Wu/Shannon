# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import datetime
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from DataAPI import convert_to_datetime, get_last_trading_day, get_trading_days, get_universe, get_next_trading_day, \
    get_stock_prices
from tqdm import tqdm

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)
output_path = r'/home/DataFolder/Stock/Derivatives/RetData/Minute'
path_base_daily = r'/home/DataFolder'
path_stock_min_bar = os.path.join(path_base_daily, 'Stock', 'Min_bars')
time_list_close = [930, 935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055,
                   1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330, 1335, 1340, 1345, 1350,
                   1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445, 1450, 1455, 1500]
time_list = [935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055,
             1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330, 1335, 1340, 1345, 1350,
             1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445, 1450, 1455, 1500]
frequency = 5
# max_workers = mp.cpu_count()
max_workers = 24
columns_used = ['timestamp', 'ticker', '5m_ret', '15m_ret', '30m_ret', '60m_ret', '120m_ret', '1d_ret', '2d_ret',
                '3d_ret', '4d_ret', '5d_ret', '10d_ret', '20d_ret']


def _get_stock_min_bar(path):
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        print(f'path:{path} is missing')
        return None


def get_stock_min_data(start_date=None
                       , end_date=None
                       , freq='1min'
                       , fq: bool = False):
    start_date = convert_to_datetime(start_date)
    if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
        end_date = convert_to_datetime(end_date)
        end_date_target = get_last_trading_day(end_date.date())
    else:
        end_date = convert_to_datetime(end_date)
        end_date_target = end_date.date()
    date_list = get_trading_days(start_date=start_date.date(), end_date=end_date_target, output='datetime')

    if fq:
        fq_string = 'fq'
    else:
        fq_string = 'raw'
    input_list = [(os.path.join(path_stock_min_bar, fq_string, '{}'.format(str(freq)),
                                str(item.year), '{:0>2}'.format(item.month),
                                datetime.date.strftime(item, '%Y%m%d') + '.par'),) for item in
                  date_list]
    group_sliced_iter = iter(input_list)

    df_list = []
    pool = mp.Pool(processes=max_workers)
    result_list = [pool.apply_async(_get_stock_min_bar, args=(*next(group_sliced_iter),))
                   for _ in range(min(max_workers, len(input_list)))]
    flag = 1
    while len(result_list) > 0:
        time.sleep(0.0001)
        status = np.array(list(map(lambda x: x.ready(), result_list)))
        if any(status):
            index = np.where(status == True)[0].tolist()
            count_flag = 0
            while index:
                out_index = index.pop(0) - count_flag
                temp = result_list[out_index].get()
                if temp is not None:
                    df_list.append(temp)
                result_list.pop(out_index)
                count_flag += 1
                if flag == 1:
                    try:
                        result_list.append(
                            pool.apply_async(_get_stock_min_bar, args=(*next(group_sliced_iter),)))
                    except StopIteration:
                        flag = 0
    pool.terminate()

    if df_list:
        df = pd.concat(df_list)
        df = df.loc[(df.index.get_level_values(0) >= start_date) & (df.index.get_level_values(0) <= end_date)]
        df.sort_index(inplace=True)
        return df
    else:
        return pd.DataFrame()


def handle_0930_df(df):
    days = df.index.date
    for day in days:
        df.loc[pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=35), :] = \
            df.loc[pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=35)] + \
            df.loc[pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=30)]
    return df.loc[~((df.index.hour == 9) & (df.index.minute == 30))]


def handler_close(data):
    df_info, return_period, days = data
    df_target_temp, df_fq, = df_info

    # df join
    df_target_temp.columns = df_target_temp.columns.get_level_values(1)
    df_join = df_target_temp.stack(dropna=False).to_frame(name='close')
    df_join['date'] = pd.to_datetime(df_join.index.get_level_values(0)).strftime('%Y%m%d').astype(int)
    df_join = df_join.reset_index().set_index(['date', 'ticker'])

    # join close
    df_join_close = df_join.join(df_fq, how='left', lsuffix='_min', rsuffix='_daily')
    df_join_close['close_min'] = df_join_close['close_min']. \
        fillna(df_join_close['close_daily'] * df_join_close['fq_factor'])
    df_join_close = df_join_close.reset_index().drop(columns=['close_daily', 'date', 'fq_factor']).set_index(
        ['timestamp', 'ticker'])
    df_close = df_join_close['close_min'].unstack()
    df_close.fillna(method='ffill', inplace=True)

    df_return = df_close.pct_change(periods=return_period).shift(-return_period)
    df_return = df_return.loc[(df_return.index >= pd.Timestamp(days[0])) &
                              (df_return.index <= pd.Timestamp(get_next_trading_day(days[-1])))]
    df_return = df_return.stack().to_frame()
    if return_period * frequency >= 245:
        df_return.columns = [f'{return_period * frequency // 245}d_ret']
    else:
        df_return.columns = [f'{return_period * frequency}m_ret']

    if df_return.dropna().empty:
        return return_period, None
    else:
        df_return.reset_index(inplace=True)
        return return_period, df_return


def handler_open(data):
    df_info, return_period, days = data
    df_target_temp, df_fq, = df_info

    # df join
    df_target_temp.columns = df_target_temp.columns.get_level_values(1)
    df_join = df_target_temp.stack(dropna=False).to_frame(name='close')
    df_join['date'] = pd.to_datetime(df_join.index.get_level_values(0)).strftime('%Y%m%d').astype(int)
    df_join = df_join.reset_index().set_index(['date', 'ticker'])

    # join close
    df_join_close = df_join.join(df_fq, how='left', lsuffix='_min', rsuffix='_daily')
    df_join_close['close_min'] = df_join_close['close_min']. \
        fillna(df_join_close['close_daily'] * df_join_close['fq_factor'])
    df_join_close = df_join_close.reset_index().drop(columns=['close_daily', 'date', 'fq_factor']).set_index(
        ['timestamp', 'ticker'])
    df_close = df_join_close['close_min'].unstack()
    df_close.fillna(method='ffill', inplace=True)

    df_return = df_close.pct_change(periods=return_period).shift(-return_period)
    df_return = df_return.loc[(df_return.index >= pd.Timestamp(days[0])) &
                              (df_return.index <= pd.Timestamp(get_next_trading_day(days[-1])))]
    df_return = df_return.stack().to_frame()
    if return_period * frequency >= 240:
        df_return.columns = [f'{return_period * frequency // 240}d_ret']
    else:
        df_return.columns = [f'{return_period * frequency}m_ret']

    if df_return.dropna().empty:
        return return_period, None
    else:
        df_return.reset_index(inplace=True)
        return return_period, df_return


def handler_vwap(data):
    df_info, return_period, days = data
    df_volume, df_amount, df_close, df_fq, period = df_info
    df_volume.fillna(0, inplace=True)
    df_amount.fillna(0, inplace=True)
    df_volume_sum = df_volume.rolling(period).sum()
    df_amount_sum = df_amount.rolling(period).sum()
    df_volume_sum.columns = df_volume_sum.columns.get_level_values(1)
    df_amount_sum.columns = df_amount_sum.columns.get_level_values(1)
    df_close.columns = df_close.columns.get_level_values(1)
    df_join = df_close.stack(dropna=False).to_frame(name='close')
    df_join['date'] = pd.to_datetime(df_join.index.get_level_values(0)).strftime('%Y%m%d').astype(int)
    df_join = df_join.reset_index().set_index(['date', 'ticker'])

    # join close
    df_join_close = df_join.join(df_fq, how='left', lsuffix='_min', rsuffix='_daily')
    df_join_close['close_min'] = df_join_close['close_min'].fillna(df_join_close['close_daily'])
    df_join_close = df_join_close.reset_index().drop(columns=['close_daily', 'date']).set_index(['timestamp', 'ticker'])
    df_fq = df_join_close['fq_factor'].unstack()
    df_close = df_join_close['close_min'].unstack()

    df_fq.fillna(method='ffill', inplace=True)
    df_close.fillna(method='ffill', inplace=True)

    # possible 0 / 0
    df_vwap = df_amount_sum / df_volume_sum
    df_vwap = df_vwap.shift(-period).fillna(df_close)
    df_vwap = df_vwap * df_fq
    df_return = df_vwap.pct_change(periods=return_period).shift(-return_period)
    df_return = df_return.loc[(df_return.index >= pd.Timestamp(days[0])) &
                              (df_return.index <= pd.Timestamp(get_next_trading_day(days[-1])))]
    df_return = df_return.stack().to_frame()
    if return_period * frequency >= 240:
        df_return.columns = [f'{return_period * frequency // 240}d_ret']
    else:
        df_return.columns = [f'{return_period * frequency}m_ret']

    if df_return.dropna().empty:
        return return_period, None
    else:
        df_return.reset_index(inplace=True)
        return return_period, df_return


def handler_output(data):
    date, df, method, tp = data
    stock_list = get_universe(date, universe='All').index.get_level_values(1)
    df_index = pd.MultiIndex.from_product([df['timestamp'].unique(), stock_list],
                                          names=('timestamp', 'ticker'))
    df = df.set_index(['timestamp', 'ticker']).reindex(df_index)
    df.drop(columns=['date'], inplace=True)
    df['trading_status'].fillna(3, inplace=True)
    df['trading_status'] = np.where(np.logical_and(np.logical_and(
        df['trading_status'] == 2, df.index.get_level_values(0).hour == 9), df.index.get_level_values(0).minute == 30),
        0, df['trading_status'])

    for column in df.columns:
        if column == 'trading_status':
            continue
        df[column] = np.where(np.isinf(df[column]), np.nan, df[column])
        if len(df[~df[column].isna()]) >= 1:
            df[column].fillna(0, inplace=True)

    df['trading_status'] = df['trading_status'].astype(int)

    if df.dropna().empty:
        return date

    date_format = datetime.datetime.strptime(date, '%Y%m%d')
    folder_year = os.path.join(output_path, '{}min'.format(frequency), method, 'tp' + str(tp), str(date_format.year),
                               '{:0>2}'.format(str(date_format.month)))
    if not os.path.exists(folder_year): os.makedirs(folder_year, exist_ok=True)
    file_name = str(date) + '.par'
    path_out_new = os.path.join(folder_year, file_name)
    df.to_parquet(path_out_new)
    return date


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def get_return_data(df, trade_days, method, forced=False, period=0, df_close_all=None):
    # Parallel
    pool = mp.Pool(processes=max_workers)

    df_trading_status = df.reindex(columns=['trading_status']).reset_index()
    if method == 'close':
        handler = handler_close
        df_daily = df_close_all
        df_close_min = df.reindex(columns=['close']).unstack().sort_index()
        days_return = 240 // frequency + 1
        return_list = [1, 3, 6, 12, 24, days_return * 1, days_return * 2, days_return * 3, days_return * 4,
                       days_return * 5, days_return * 10, days_return * 20]
        df_target = (df_close_min, df_daily)

    elif method == 'open':
        handler = handler_open
        df_daily = df_close_all
        df_target = df.reindex(columns=['open']).unstack().sort_index()
        df_target = df_target.loc[~((df_target.index.hour == 9) &
                                    (df_target.index.minute == 30))]
        days_return = 240 // frequency
        return_list = [1, 3, 6, 12, 24, days_return * 1, days_return * 2, days_return * 3, days_return * 4,
                       days_return * 5, days_return * 10, days_return * 20]
        df_target = (df_target, df_daily)

    elif method == 'vwap':
        handler = handler_vwap
        df_volume = df.reindex(columns=['volume']).unstack().sort_index()
        df_amount = df.reindex(columns=['amount']).unstack().sort_index()
        df_close, df_daily = df_close_all
        df_close = df_close.reindex(columns=['close']).unstack().sort_index()
        df_close = df_close.loc[~((df_close.index.hour == 9) & (df_close.index.minute == 30))]
        df_volume = handle_0930_df(df_volume)
        df_amount = handle_0930_df(df_amount)
        days_return = 240 // frequency
        return_list = [1, 3, 6, 12, 24, days_return * 1, days_return * 2, days_return * 3, days_return * 4,
                       days_return * 5, days_return * 10, days_return * 20]
        df_target = (df_volume, df_amount, df_close, df_daily, period)

    else:
        raise ValueError(f'Unsupported method: {method}')

    # TASK1: get return data
    item_para = [(df_target, return_period, trade_days) for return_period in return_list]
    stock_iter = iter(item_para)
    stock_batch_num = len(item_para)
    result_list = [pool.apply_async(handler, args=(next(stock_iter),))
                   for _ in range(min(max_workers, stock_batch_num))]
    flag = 1
    df_dict = dict()
    with tqdm(total=stock_batch_num, ncols=150) as pbar:
        while len(result_list) > 0:
            time.sleep(0.0001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count = 0
                while index:
                    out_index = index.pop(0) - count
                    periods, df = result_list[out_index].get()
                    if df is not None:
                        df_dict[periods] = df
                    result_list.pop(out_index)
                    count += 1
                    pbar.set_description(f"Handling return data | period {periods}...")
                    pbar.update(1)
                    if flag == 1:
                        try:
                            result_list.append(
                                pool.apply_async(handler, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0

    index_list = sorted(df_dict)
    df_list = [df_dict[idx] for idx in index_list]
    if forced:
        assert len(df_list) == len(return_list), f"mismatch length: " \
                                                 f"df_list {len(df_list)} vs return_list: {len(return_list)}"

    if len(df_list) == 0:
        print("No data to create")
        return

    elif len(df_list) == 1:
        df_target = df_list[0]
    else:
        df_target = df_list[0]
        for item in df_list[1:]:
            df_target = df_target.merge(item, on=['timestamp', 'ticker'], how='left', copy=False)
    df_target = df_target.reindex(columns=columns_used)
    df_target = df_target.merge(df_trading_status, on=['timestamp', 'ticker'], how='left', copy=False)

    # TAKS2: output daily data
    df_target['date'] = df_target['timestamp'].dt.strftime('%Y%m%d')
    grouped = df_target.groupby('date')
    item_para = [(date, group, method, period) for date, group in grouped]
    stock_iter = iter(item_para)
    stock_batch_num = len(item_para)
    result_list = [pool.apply_async(handler_output, args=(next(stock_iter),))
                   for _ in range(min(max_workers, stock_batch_num))]
    flag = 1
    with tqdm(total=stock_batch_num, ncols=150) as pbar:
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
                    pbar.set_description(f"output return data of {date}...")
                    pbar.update(1)
                    if flag == 1:
                        try:
                            result_list.append(
                                pool.apply_async(handler_output, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0
    pool.terminate()


def run():
    # initial start: 2015-01-01
    start_date, end_date = '2015-01-01', '2021-08-10'
    method_list = ['vwap']  # support open, close, twap, vwap
    day_length = 125
    periods = [24]
    days_count = 20

    assert frequency != 1 and isinstance(frequency, int), \
        'frequency cannot be assign with value: {}'.format(frequency)

    # create
    forced = False
    # # update
    # forced = False
    trade_days = get_trading_days(start_date=start_date, end_date=end_date, output='datetime')

    num_list = make_batches(len(trade_days), day_length)
    for start_temp, end_temp in num_list:
        trade_day_list = trade_days[start_temp:end_temp]
        print('>>> generate min bar data, date range: from {} to {}.'.format(trade_day_list[0], trade_day_list[-1]))
        if 'vwap' in method_list or 'twap' in method_list:
            wap_day_gap = max(periods) // 48
        else:
            wap_day_gap = 0
        df_all = get_stock_min_data(start_date=trade_day_list[0],
                                    end_date=get_trading_days(
                                        start_date=get_next_trading_day(trade_day_list[-1]),
                                        count=days_count + 1 + wap_day_gap)[
                                        -1],
                                    freq=str(frequency) + 'min', fq=True). \
            reindex(columns=['open', 'close', 'twap', 'volume', 'amount', 'trading_status'])

        if 'vwap' in method_list:
            df_all_close = get_stock_min_data(start_date=trade_day_list[0],
                                              end_date=get_trading_days(
                                                  start_date=get_next_trading_day(trade_day_list[-1]),
                                                  count=days_count + 1 + wap_day_gap)[
                                                  -1],
                                              freq=str(frequency) + 'min', fq=False). \
                reindex(columns=['close'])

        df_daily_close_fq = get_stock_prices(start_date=trade_day_list[0],
                                             end_date=get_trading_days(
                                                 start_date=get_next_trading_day(trade_day_list[-1]),
                                                 count=days_count + 1 + wap_day_gap)[
                                                 -1],
                                             freq='daily', fq=False). \
            reindex(columns=['close', 'fq_factor'])
        df_daily_close_fq = df_daily_close_fq.rename_axis(['date', 'ticker'])

        # # eval
        # df_all = df_all.loc[df_all.index.get_level_values(1) == '002509']
        # df_all_close = df_all_close.loc[df_all_close.index.get_level_values(1) == '002509']

        if df_all.empty:
            print('No need to calculate, no data.')
            return

        for method in method_list:
            if method in ('open', 'close'):
                get_return_data(df_all, trade_day_list, method, forced=forced,
                                df_close_all=df_daily_close_fq)
            else:
                for period in periods:
                    get_return_data(df_all, trade_day_list, method, forced=forced, period=period,
                                    df_close_all=(df_all_close, df_daily_close_fq,))


if __name__ == '__main__':
    run()
