import datetime
import multiprocessing as mp
import os
import time

import DataAPI as api
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

max_workers = mp.cpu_count()
time_list = []
frequency = 1
min_return_path = f'/home/DataFolder/Stock/Derivatives/RetData/Minute/{frequency}min'
output_path = f'/home/DataFolder/Stock/Derivatives/RetData/Daily/{frequency}min'
i = 930
while i <= 1130:
    if i % 100 == 60:
        i -= 60
        i += 100
    time_list.append(i)
    i += frequency

i = 1301
while i <= 1500:
    if i % 100 == 60:
        i -= 60
        i += 100
    time_list.append(i)
    i += frequency


def handler(input_set):
    day, method, ti, tp = input_set
    target_path = os.path.join(min_return_path, method, 'tp' + str(tp), str(day.year),
                               '{:0>2}'.format(str(day.month)), datetime.datetime.strftime(day, '%Y%m%d') + '.par')
    df = pd.read_parquet(target_path)
    if method == 'close':
        time_list_target = time_list
    else:
        time_list_target = time_list
    time_target = time_list_target[ti]
    df_target = df.loc[(df.index.get_level_values(0).hour == time_target // 100) &
                       (df.index.get_level_values(0).minute == time_target % 100)]
    df_target.reset_index(inplace=True)
    if method in ('open', 'twap', 'vwap'):
        df_target['timestamp'] = df_target['timestamp'] - pd.Timedelta(minutes=5)
    df_target.set_index(['timestamp', 'ticker'], inplace=True)
    for column in df_target.columns:
        df_target[column] = np.where(np.isinf(df_target[column]), np.nan, df_target[column])
    df_target['trading_status'] = df_target['trading_status'].astype(int)
    output_path_target = os.path.join(output_path, method, 'ti' + str(ti), 'tp' + str(tp), str(day.year),
                                      '{:0>2}'.format(str(day.month)),
                                      datetime.datetime.strftime(day, '%Y%m%d') + '.par')

    folder = os.path.dirname(output_path_target)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df_target.to_parquet(output_path_target)
    return day


def convert(start_date, end_date, method, ti, tp=0):
    days = api.get_trading_days(start_date, end_date)
    if method in ('open', 'close'):
        assert tp == 0, "when method is `open` or `close`, parameter tp should be 0."
        item_para = [(day, method, ti, tp) for day in days]
    else:
        assert tp > 0, "when method is `twap` or `vwap`, parameter tp should be greater than 0."
        item_para = [(day, method, ti, tp) for day in days]

    pool = mp.Pool(processes=max_workers)
    stock_iter = iter(item_para)
    stock_batch_num = len(item_para)
    result_list = [pool.apply_async(handler, args=(next(stock_iter),))
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
                    day = result_list[out_index].get()
                    result_list.pop(out_index)
                    count += 1
                    pbar.set_description(f"Handling return dat0a: {day}...")
                    pbar.update(1)
                    if flag == 1:
                        try:
                            result_list.append(
                                pool.apply_async(handler, args=(next(stock_iter),)))
                        except StopIteration:
                            flag = 0


if __name__ == '__main__':
    start, end = '2021-11-01', '2021-12-10'
    price_methods = ['vwap']
    time_start = 0
    trading_period = 31
    for price_method in price_methods:
        convert(start, end, price_method, time_start, tp=trading_period)
