# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import datetime
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
from DataAPI import convert_to_datetime, get_last_trading_day, get_trading_days, get_universe
from tqdm import tqdm

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)
output_path = r'/home/DataFolder/Stock/Derivatives/RetData/Daily'
path_base_daily = r'/home/DataFolder'
path_stock_min_bar = os.path.join(path_base_daily, 'Stock', 'Min_bars')
max_workers = mp.cpu_count()
time_list_close = [930, 935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055,
                   1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330, 1335, 1340, 1345, 1350,
                   1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445, 1450, 1455, 1500]
time_list = [935, 940, 945, 950, 955, 1000, 1005, 1010, 1015, 1020, 1025, 1030, 1035, 1040, 1045, 1050, 1055,
             1100, 1105, 1110, 1115, 1120, 1125, 1130, 1305, 1310, 1315, 1320, 1325, 1330, 1335, 1340, 1345, 1350,
             1355, 1400, 1405, 1410, 1415, 1420, 1425, 1430, 1435, 1440, 1445, 1450, 1455, 1500]


def _get_stock_min_bar(path):
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
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


def handler_close(data, trading_time):
    date, df_temp = data
    df_target_temp = df_temp['close'].unstack().sort_index()
    df_all_status = df_temp['trading_status']

    days_return = 240 // frequency + 1
    return_list = [1, 3, 6, 12, 24, days_return * 1, days_return * 2, days_return * 3, days_return * 4, days_return * 5]
    return_list.reverse()
    df_return = df_target_temp.pct_change(periods=return_list[0]).shift(-return_list[0]).stack().to_frame(name='5d_ret')

    for i in return_list[1:]:
        name = '{}th_ret'.format(str(i))
        df_temp = df_target_temp.pct_change(periods=i).shift(-i).stack().to_frame(name=name)
        df_return = df_return.join(df_temp, how='right')

    df_return.columns = ['5d_ret', '4d_ret', '3d_ret', '2d_ret', '1d_ret', f'{24 * frequency}m_ret',
                         f'{12 * frequency}m_ret', f'{6 * frequency}m_ret', f'{3 * frequency}m_ret',
                         f'{1 * frequency}m_ret']
    target_columms = ['5d_ret', '4d_ret', '3d_ret', '2d_ret', '1d_ret', f'{24 * frequency}m_ret',
                      f'{12 * frequency}m_ret', f'{6 * frequency}m_ret', f'{3 * frequency}m_ret',
                      f'{1 * frequency}m_ret']
    target_columms.reverse()
    df_return = df_return.reindex(columns=target_columms)
    df_return = df_return.join(df_all_status, how='left')
    minute_target = time_list_close[ti] % 100
    hour_target = time_list_close[ti] // 100
    df_return = df_return.loc[(df_return.index.get_level_values(0).hour == hour_target) &
                              (df_return.index.get_level_values(0).minute == minute_target)]
    df_return.reset_index(inplace=True)
    df_return['date'] = df_return['timestamp'].dt.strftime('%Y%m%d').astype(int)
    df_return = df_return.loc[df_return['date'] == int(datetime.date.strftime(date, "%Y%m%d"))]
    df_return.set_index(['date', 'ticker'], inplace=True)
    df_return.sort_index(inplace=True)
    df_universe = get_universe(date, universe='All').rename_axis(index={'timestamp': 'date'})
    df_return = df_return.join(df_universe, how='left').reset_index().set_index(['timestamp', 'ticker'])

    df_return.drop(columns=['stock_name', 'ipo_date', 'enlist_date', 'tradable', 'date'], inplace=True)
    df_return['trading_status'].fillna(2, inplace=True)
    df_return['trading_status'] = df_return['trading_status'].astype(int)

    if trading_time == 0:
        df_return['trading_status'] = np.where(df_return['trading_status'] == 2, 0, df_return['trading_status'])
    # df_return = df_return.reset_index().drop(columns='timestamp').set_index('ticker')
    if forced:
        df_return = df_return.fillna(0)
    else:
        if df_return.dropna(subset=['1d_ret'], how='all').empty:
            return

    folder_year = os.path.join(output_path, '{}min'.format(frequency), 'close', 'ti' + str(trading_time), 'tp0',
                               str(date.year), '{:0>2}'.format(str(date.month)))
    if not os.path.exists(folder_year): os.makedirs(folder_year, exist_ok=True)
    file_name = datetime.datetime.strftime(date, '%Y%m%d') + '.par'
    path_out_new = os.path.join(folder_year, file_name)
    df_return.to_parquet(path_out_new)
    return date


def handler_open(data, trading_time):
    date, df_temp = data
    df_temp = df_temp.loc[~((df_temp.index.get_level_values(0).hour == 9) &
                            (df_temp.index.get_level_values(0).minute == 30))]
    minute_target = time_list[ti] % 100
    hour_target = time_list[ti] // 100
    df_target_temp = df_temp['open'].unstack().sort_index()
    df_all_status = df_temp['trading_status']

    days_return = 240 // frequency
    return_list = [1, 3, 6, 12, 24, days_return * 1, days_return * 2, days_return * 3, days_return * 4, days_return * 5]
    return_list.reverse()
    df_return = df_target_temp.pct_change(periods=return_list[0]).shift(-return_list[0]).stack().to_frame(name='5d_ret')

    for i in return_list[1:]:
        name = '{}th_ret'.format(str(i))
        df_temp = df_target_temp.pct_change(periods=i).shift(-i).stack().to_frame(name=name)
        df_return = df_return.join(df_temp, how='right')

    df_return.columns = ['5d_ret', '4d_ret', '3d_ret', '2d_ret', '1d_ret', f'{24 * frequency}m_ret',
                         f'{12 * frequency}m_ret', f'{6 * frequency}m_ret', f'{3 * frequency}m_ret',
                         f'{1 * frequency}m_ret']
    target_columms = ['5d_ret', '4d_ret', '3d_ret', '2d_ret', '1d_ret', f'{24 * frequency}m_ret',
                      f'{12 * frequency}m_ret', f'{6 * frequency}m_ret', f'{3 * frequency}m_ret',
                      f'{1 * frequency}m_ret']
    target_columms.reverse()
    df_return = df_return.reindex(columns=target_columms)
    df_return = df_return.join(df_all_status, how='left')

    df_return = df_return.loc[(df_return.index.get_level_values(0).hour == hour_target) &
                              (df_return.index.get_level_values(0).minute == minute_target)]
    df_return.reset_index(inplace=True)
    df_return['date'] = df_return['timestamp'].dt.strftime('%Y%m%d').astype(int)
    df_return = df_return.loc[df_return['date'] == int(datetime.date.strftime(date, "%Y%m%d"))]
    df_return.set_index(['date', 'ticker'], inplace=True)
    df_return.sort_index(inplace=True)
    df_universe = get_universe(date, universe='All').rename_axis(index={'timestamp': 'date'})
    df_return = df_return.join(df_universe, how='left').reset_index().set_index(['timestamp', 'ticker'])
    df_return.drop(columns=['stock_name', 'ipo_date', 'enlist_date', 'tradable', 'date'], inplace=True)
    df_return['trading_status'].fillna(2, inplace=True)
    df_return['trading_status'] = df_return['trading_status'].astype(int)

    # df_return = df_return.reset_index().drop(columns='timestamp').set_index('ticker')
    if forced:
        df_return = df_return.fillna(0)
    else:
        if df_return.dropna(subset=['1d_ret'], how='all').empty:
            return

    folder_year = os.path.join(output_path, '{}min'.format(frequency), 'open', 'ti' + str(trading_time), 'tp0',
                               str(date.year), '{:0>2}'.format(str(date.month)))
    if not os.path.exists(folder_year): os.makedirs(folder_year, exist_ok=True)
    file_name = datetime.datetime.strftime(date, '%Y%m%d') + '.par'
    path_out_new = os.path.join(folder_year, file_name)
    df_return.to_parquet(path_out_new)
    return date


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


def get_return_data(trade_days_list, trading_time, df, forced=False):
    # Parallel
    pool = mp.Pool(processes=max_workers)
    if forced:
        assert len(trade_days_list) >= 9, "Length is smaller than 9, cannot calculate! check again!"
        trading_days = [trade_days_list[i:i + 8] for i in range(len(trade_days_list) - 7)]
    else:
        trading_days = [trade_days_list[i:i + 8] for i in range(len(trade_days_list))]
    trading_day_target = [(item[0],
                           df.loc[(df.index.get_level_values(0) >= pd.Timestamp(item[0]))
                                  & (df.index.get_level_values(0) < pd.Timestamp(item[-1]) + pd.Timedelta(days=1))])
                          for item in trading_days]

    stock_iter = iter(trading_day_target)
    stock_batch_num = len(trading_day_target)

    if method == 'close':
        handler = handler_close
    elif method == 'open':
        handler = handler_open
    else:
        raise ValueError(f'Unsupported method: {method}')

    # TASK1: load data
    result_list = [pool.apply_async(handler, args=(next(stock_iter), trading_time,))
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
                    pbar.set_description("Loading daily data {} | ti {}...".format(str(date), str(ti)))
                    pbar.update(1)
                    if flag == 1:
                        try:
                            result_list.append(
                                pool.apply_async(handler, args=(next(stock_iter), trading_time,)))
                        except StopIteration:
                            flag = 0


if __name__ == '__main__':
    frequency = 5
    ti_list = [0]
    tp_list = [0]
    method = 'close'
    day_length = 100

    assert frequency != 1 and isinstance(frequency, int), \
        'frequency cannot be assign with value: {}'.format(frequency)

    # create
    # forced = True
    # update
    forced = False
    trade_days = get_trading_days(start_date='2021-04-01', end_date='2021-08-02', output='datetime')

    num_list = make_batches(len(trade_days), day_length)
    for start_temp, end_temp in num_list:
        trade_day_list = trade_days[start_temp:end_temp]
        print('>>> generate date range: from {} to {}.'.format(trade_day_list[0], trade_day_list[-1]))
        df_all = get_stock_min_data(start_date=trade_day_list[0],
                                    end_date=get_trading_days(start_date=trade_day_list[-1], count=8)[-1],
                                    freq=str(frequency) + 'min', fq=True)

        for ti in ti_list:
            get_return_data(trade_day_list, ti, df_all, forced=forced)
