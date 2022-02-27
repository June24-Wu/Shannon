# !/usr/bin/python3.7
# -*- coding: UTF-8 -*-
# @author: guichuan
import pandas as pd
import os
import datetime
import DataAPI as api
import numpy as np
import multiprocessing as mp


def convert_to_standard_daily_bt_csv(combo_name: str, df: pd.DataFrame, output_path: str):
    columns = df.columns.tolist()
    for item in ('symbol', 'timestamp', 'score'):
        assert item in columns, f"{item} not in the dataframe. Check again!"

    dtype_time = df['timestamp'].dtype
    assert dtype_time == int, f"Timestamp of dataframe should be int, but {dtype_time} received!"
    grouped = df[['symbol', 'timestamp', 'score']].groupby('timestamp')
    for date, group in grouped:
        assert api.is_trading_day(date), f"{date} is not a trading date!"
        date_format = datetime.datetime.strptime(str(date), '%Y%m%d').date()
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.csv'
        folder = os.path.join(output_path, combo_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file_name)
        group.drop(columns='timestamp', inplace=True)
        group.sort_values(by='score', ascending=False, inplace=True)
        group.to_csv(file, sep=',', mode='w', header=False, index=False, encoding='utf-8')


def convert_to_standard_daily_bt_par(combo_name: str, df: pd.DataFrame, output_path: str):
    columns = df.columns.tolist()
    for item in ('symbol', 'timestamp', 'score'):
        assert item in columns, f"{item} not in the dataframe. Check again!"

    dtype_time = df['timestamp'].dtype
    assert dtype_time == int, f"Timestamp of dataframe should be int, but {dtype_time} received!"
    grouped = df[['symbol', 'timestamp', 'score']].groupby('timestamp')
    for date, group in grouped:
        assert api.is_trading_day(date), f"{date} is not a trading date!"
        date_format = datetime.datetime.strptime(str(date), '%Y%m%d').date()
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.par'
        folder = os.path.join(output_path, combo_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file_name)
        group.drop(columns='timestamp', inplace=True)
        group.set_index('symbol', inplace=True)
        group.index.name = 'ticker'
        group.sort_values(by='score', ascending=False, inplace=True)
        group.to_parquet(file)


def convert_to_standard_daily_feature_csv(alpha_name: str, df: pd.DataFrame, output_path: str):
    columns = df.columns.tolist()
    for item in ('symbol', 'timestamp', alpha_name):
        assert item in columns, f"{item} not in the dataframe. Check again!"

    dtype_time = df['timestamp'].dtype
    assert dtype_time == np.dtype('datetime64[ns]'), \
        f"Timestamp of dataframe should be pd.Timestamp, but {dtype_time} received!"
    temp = df.set_index('timestamp')
    assert temp.between_time('9:30', '11:30').shape[0] + temp.between_time('13:00', '15:00').shape[0] == df.shape[0], \
        "Non-trading timestamp exsist!"

    grouped = df[['timestamp', 'symbol', alpha_name]].groupby('timestamp')
    for date, group in grouped:
        date_format = pd.to_datetime(date).date()
        assert api.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.csv'
        folder = os.path.join(output_path, alpha_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file_name)
        group.sort_values(by=alpha_name, ascending=False, inplace=True)
        group.to_csv(file, sep=',', mode='w', header=False, index=False, encoding='utf-8')


def convert_to_standard_daily_feature_par(alpha_name: str, df: pd.DataFrame, output_path: str):
    columns = df.columns.tolist()
    for item in ('symbol', 'timestamp', alpha_name):
        assert item in columns, f"{item} not in the dataframe. Check again!"

    dtype_time = df['timestamp'].dtype
    assert dtype_time == np.dtype('datetime64[ns]'), \
        f"Timestamp of dataframe should be pd.Timestamp, but {dtype_time} received!"
    temp = df.set_index('timestamp')
    assert temp.between_time('9:30', '11:30').shape[0] + temp.between_time('13:00', '15:00').shape[0] == df.shape[0], \
        "Non-trading timestamp exsist!"
    df.rename(columns={'symbol': 'ticker'}, inplace=True)

    grouped = df[['timestamp', 'ticker', alpha_name]].groupby('timestamp')
    for date, group in grouped:
        date_format = pd.to_datetime(date).date()
        assert api.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.par'
        folder = os.path.join(output_path, alpha_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)

        file = os.path.join(folder, file_name)
        group.set_index(['timestamp', 'ticker'], inplace=True)
        group.sort_values(by=alpha_name, ascending=False, inplace=True)
        group.to_parquet(file)


def _read_csv(input_data):
    path, date = input_data
    df = pd.read_csv(path, header=None, dtype={0: str}, names=['symbol', 'score'])
    df.insert(0, 'timestamp', date)
    return df


def convert_bt_csv_to_dataframe(execute_path):
    file_list = []
    for curDir, dirs, files in os.walk(execute_path):
        if files:
            for file in files:
                file_list.append((os.path.join(curDir, file), int(file.split('.')[0])))

    pool = mp.Pool(mp.cpu_count())
    result_list = pool.map(_read_csv, file_list)
    pool.close()
    pool.join()

    df = pd.concat(result_list)
    return df


if __name__ == '__main__':
    execute_path = r'/home/ShareFolder/lgc/data/demo_data/platform/backtest/csv'
    df = convert_bt_csv_to_dataframe(execute_path)
    print(df.head())
