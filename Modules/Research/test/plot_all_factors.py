import multiprocessing as mp
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Research.database.mysql import MysqlAPI
from tqdm import tqdm

matplotlib.use('TkAgg')
max_workers = 24
MYSQL_INFO = {
    'host': '172.16.1.13',
    'port': 3306,
    'user': 'dev_liguichuan',
    'password': '!*EeJZ5I4O8gdzPb8m-+',
    'db': 'factor_lib_ti0',
    'charset': 'utf8'
}
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)
dapi = MysqlAPI(MYSQL_INFO)
start, end = '2021-01-01', '2021-06-30'
# alpha_list = dapi.get_factor_info().table_name.tolist()
alpha_list = ['gp_liguichuan_alpha_00000_20150101_20180630']
path_output = f'/home/ShareFolder/lgc/data/factor_lib_ti0/plot'
if not os.path.exists(path_output):
    os.makedirs(path_output)


def handler(alpha_name):
    api = MysqlAPI(MYSQL_INFO)
    df = api.query_factor_ic(alpha_name, start_time=start, end_time=end, period=('1d', '2d', '3d', '4d', '5d'))
    x_data = df.trading_time.tolist()
    y_data_1 = df.IC_1d
    y_data_2 = df.IC_2d
    y_data_3 = df.IC_3d
    y_data_4 = df.IC_4d
    y_data_5 = df.IC_5d
    y_data_sum_1 = y_data_1.cumsum() * 0.01
    y_data_sum_2 = y_data_2.cumsum() * 0.01
    y_data_sum_3 = y_data_3.cumsum() * 0.01
    y_data_sum_4 = y_data_4.cumsum() * 0.01
    y_data_sum_5 = y_data_5.cumsum() * 0.01
    x_data_slice = x_data[60:]

    fig1 = plt.figure(f'Figure_{alpha_name}', figsize=(20, 10)).add_subplot(221)
    y_data_sum_1_slice = y_data_sum_1.iloc[60:]
    y_data_sum_2_slice = y_data_sum_2.iloc[60:]
    y_data_sum_3_slice = y_data_sum_3.iloc[60:]
    y_data_sum_4_slice = y_data_sum_4.iloc[60:]
    y_data_sum_5_slice = y_data_sum_5.iloc[60:]
    fig1.plot(x_data_slice, y_data_sum_1_slice, label='IC_1d')
    fig1.plot(x_data_slice, y_data_sum_2_slice, label='IC_2d')
    fig1.plot(x_data_slice, y_data_sum_3_slice, label='IC_3d')
    fig1.plot(x_data_slice, y_data_sum_4_slice, label='IC_4d')
    fig1.plot(x_data_slice, y_data_sum_5_slice, label='IC_5d')
    fig1.set_ylabel('IC_cum')
    fig1.legend()
    fig1.set_title(f'{alpha_name}')

    fig2 = plt.figure(f'Figure_{alpha_name}', figsize=(20, 10)).add_subplot(222)
    fig2.plot(x_data_slice, y_data_1.rolling(60).mean().iloc[60:], label='IC_1d')
    fig2.plot(x_data_slice, y_data_2.rolling(60).mean().iloc[60:], label='IC_2d')
    fig2.plot(x_data_slice, y_data_3.rolling(60).mean().iloc[60:], label='IC_3d')
    fig2.plot(x_data_slice, y_data_4.rolling(60).mean().iloc[60:], label='IC_4d')
    fig2.plot(x_data_slice, y_data_5.rolling(60).mean().iloc[60:], label='IC_5d')
    fig2.set_ylabel('IC_ma')
    fig2.legend()
    fig1.set_title(f'{alpha_name}')

    # chenyi
    # result = dict()
    # ma_1 = y_data_sum_1.rolling(60).mean()
    # ma_2 = y_data_sum_2.rolling(60).mean()
    # ma_3 = y_data_sum_3.rolling(60).mean()
    # ma_4 = y_data_sum_4.rolling(60).mean()
    # ma_5 = y_data_sum_5.rolling(60).mean()
    # for i in range(len(x_data_slice)):
    #     item = x_data_slice[i]
    #     result[item] = dict()
    #     for j in range(5):
    #         result[item][f"IC_{j}d"] = eval(f"ma_{j+1}").iloc[60 + j]

    fig3 = plt.figure(f'Figure_{alpha_name}', figsize=(20, 10)).add_subplot(223)
    fig3.plot(x_data_slice, (y_data_sum_1 / (pd.Series(range(len(y_data_sum_1))) + 1)).iloc[60:], label='IC_1d_mean')
    fig3.plot(x_data_slice, (y_data_sum_2 / (pd.Series(range(len(y_data_sum_2))) + 1)).iloc[60:], label='IC_2d_mean')
    fig3.plot(x_data_slice, (y_data_sum_3 / (pd.Series(range(len(y_data_sum_3))) + 1)).iloc[60:], label='IC_3d_mean')
    fig3.plot(x_data_slice, (y_data_sum_4 / (pd.Series(range(len(y_data_sum_4))) + 1)).iloc[60:], label='IC_4d_mean')
    fig3.plot(x_data_slice, (y_data_sum_5 / (pd.Series(range(len(y_data_sum_5))) + 1)).iloc[60:], label='IC_5d_mean')
    fig3.set_ylabel('IC_cummean')
    fig3.legend()

    df = api.query_factor_return(alpha_name, start_time=start, end_time=end)
    fig4 = plt.figure(f'Figure_{alpha_name}', figsize=(20, 10)).add_subplot(224)
    try:
        y_data_1 = (df.alpha_group0 - df.alpha_group0[60]) / df.alpha_group0[60] + 1
        y_data_2 = (df.alpha_group1 - df.alpha_group1[60]) / df.alpha_group1[60] + 1
        y_data_3 = (df.alpha_group2 - df.alpha_group2[60]) / df.alpha_group2[60] + 1
        y_data_4 = (df.alpha_group3 - df.alpha_group3[60]) / df.alpha_group3[60] + 1
        y_data_5 = (df.alpha_group4 - df.alpha_group4[60]) / df.alpha_group4[60] + 1
        y_data_6 = (df.alpha_group5 - df.alpha_group5[60]) / df.alpha_group5[60] + 1
        y_data_7 = (df.alpha_group6 - df.alpha_group6[60]) / df.alpha_group6[60] + 1
        y_data_8 = (df.alpha_group7 - df.alpha_group7[60]) / df.alpha_group7[60] + 1
        y_data_9 = (df.alpha_group8 - df.alpha_group8[60]) / df.alpha_group8[60] + 1
        y_data_10 = (df.alpha_group9 - df.alpha_group9[60]) / df.alpha_group9[60] + 1

        fig4.plot(x_data_slice, y_data_1.iloc[60:], label='group_0')
        fig4.plot(x_data_slice, y_data_2.iloc[60:], label='group_1')
        fig4.plot(x_data_slice, y_data_3.iloc[60:], label='group_2')
        fig4.plot(x_data_slice, y_data_4.iloc[60:], label='group_3')
        fig4.plot(x_data_slice, y_data_5.iloc[60:], label='group_4')
        fig4.plot(x_data_slice, y_data_6.iloc[60:], label='group_5')
        fig4.plot(x_data_slice, y_data_7.iloc[60:], label='group_6')
        fig4.plot(x_data_slice, y_data_8.iloc[60:], label='group_7')
        fig4.plot(x_data_slice, y_data_9.iloc[60:], label='group_8')
        fig4.plot(x_data_slice, y_data_10.iloc[60:], label='group_9')
        fig4.set_ylabel('return')
        fig4.legend()
    except IndexError:
        plt.close('all')
        print(alpha_name)
        return

    except ValueError:
        plt.close('all')
        print(alpha_name)
        return
    plt.show()
    plt.savefig(os.path.join(path_output, f'{alpha_name}.jpg'))
    plt.close('all')


def run():
    group_sliced_iter = iter(alpha_list)
    pool = mp.Pool(processes=max_workers)
    result_list = [pool.apply_async(handler, args=(next(group_sliced_iter),))
                   for _ in range(min(max_workers, len(alpha_list)))]
    flag = 1
    with tqdm(total=len(alpha_list), ncols=150) as pbar:
        while len(result_list) > 0:
            time.sleep(0.0001)
            status = np.array(list(map(lambda x: x.ready(), result_list)))
            if any(status):
                index = np.where(status == True)[0].tolist()
                count_flag = 0
                while index:
                    out_index = index.pop(0) - count_flag
                    result_list[out_index].get()
                    result_list.pop(out_index)
                    count_flag += 1
                    if flag == 1:
                        try:
                            result_list.append(
                                pool.apply_async(handler, args=(next(group_sliced_iter),)))
                        except StopIteration:
                            flag = 0

                    pbar.set_description("output alphas...")
                    pbar.update(1)

    pool.terminate()


if __name__ == '__main__':
    run()
