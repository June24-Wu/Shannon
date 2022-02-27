import calendar
import datetime
import time

import DataAPI as api
import utils.namespace as namespace
import pandas as pd

from Research.backtest.bt import BTDaily

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 300)


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


def get_time_list(start, end, gap):
    start = api.convert_to_datetime(start)
    end = api.convert_to_datetime(end)

    _, date_end_flag = _get_date_to_slice(start, num=gap)
    date_end_flag += datetime.timedelta(hours=15)
    result_list = []
    flag = 1
    while date_end_flag < end and flag == 1:
        next_start, date_end_flag = _get_date_to_slice(start, num=gap)

        if date_end_flag + datetime.timedelta(days=1) < end:
            result_list.append((int(start.strftime('%Y%m%d')), int(date_end_flag.strftime('%Y%m%d'))))
            start = next_start
            date_end_flag = date_end_flag + datetime.timedelta(days=1)
        else:
            result_list.append((int(start.strftime('%Y%m%d')), int(end.strftime('%Y%m%d'))))
            flag = 0

    if flag == 1:
        result_list = [int(start.strftime('%Y%m%d')), int(end.strftime('%Y%m%d'))]
    return result_list


def test_universe(universe, start, end, gap=12, weight_method='equal'):
    print(f"======== loading universe: {universe} ==========")
    df = api.get_universe_series(universe, start_date=start, end_date=end, sort=True, allow_missing=True)
    df = df.reset_index().set_index('timestamp').drop(columns=['tradable'])
    # df.fillna(1, inplace=True)
    df.index = df.index.astype(int)
    print('Universe is loaded.')

    config_path = r'../../config/combo_bt_template'
    configs = namespace.load_namespace(config_path)
    configs.trading_type = 'long-only'
    configs.data_format = 'dataframe'
    configs.ti = 0
    configs.trade_period = 24
    configs.stock_percentage = True
    configs.stock_num = 0.5
    configs.transmission_rate = 0.00025
    configs.daily_data_missing_allowed = True
    configs.benchmark = "ZZ500"
    configs.bt_price = 'vwap'
    configs.universe = "All"
    configs.weight = weight_method
    configs.change_pos_threshold = 0.5
    configs.keep_pos_percentile = 0.0

    print('=================== parameters: ===================')
    print('start_ti: ', str(configs.ti))
    print('trading_period: ', str(configs.trade_period))
    print('transmission_rate: ', str(configs.transmission_rate))
    print('tax_rate: ', str(configs.tax_rate))
    print('bt_price: ', str(configs.bt_price))
    print('Benchmark: ', str(configs.benchmark))
    print('Weight: ', str(configs.weight))
    print("=" * 50)

    bt = BTDaily(configs, start_date=start, end_date=end)

    result_list = get_time_list(start, end, gap)

    bt.feed_data(df)

    start = time.time()
    bt.run()
    df, df_final = bt.evaluate(evalRange=tuple(result_list))
    df, df_final = bt.evaluate(evalRange=((20211101, 20211202), ))

    end = time.time()
    print('Time consume: %.2f seconds\n' % (end - start))


if __name__ == '__main__':
    # task = 'FundsHeavyHoldingsIncrease_200'
    task = 'FundsHeavyHoldingsFilter'
    start_date, end_date = '2017-01-01', '2021-12-02'
    test_universe(task, start_date, end_date, gap=12, weight_method='equal')
