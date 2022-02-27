import pandas as pd
import numpy as np
import DataAPI as api
import pickle
import os
import datetime

output_path = r'/home/DataFolder/Stock/Derivatives/DataType'


class DataTypeOperator:

    @staticmethod
    def create_StockDailyPrices(start_date, end_date, fq):
        df = api.get_stock_prices(start_date=start_date, end_date=end_date, fq=fq, sort=True)
        result = dict()
        for key in df.keys():
            temp = df[key].unstack()
            result[key] = temp

        if fq:
            string = "Fq"
        else:
            string = 'Raw'
        file_path = os.path.join(output_path, f'StockDailyPrices{string}.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    @staticmethod
    def load_StockDailyPrices(fq):
        if fq:
            string = "Fq"
        else:
            string = 'Raw'
        file_path = os.path.join(output_path, f'StockDailyPrices{string}.pkl')
        with open(file_path, 'rb') as f:
            result = pickle.load(f)
        return result

    @staticmethod
    def update_StockDailyPrices():
        for fq in (True, False):
            temp = DataTypeOperator.load_StockDailyPrices(fq)
            timestamp_last = temp[list(temp.keys())[0]].index.values[-1]
            start_date = api.get_next_trading_day(timestamp_last)
            end_date = datetime.date.today()

            try:
                df = api.get_stock_prices(start_date=start_date, end_date=end_date, fq=fq, sort=True)
            except Exception as err:
                print(err)
                return

            if df.empty:
                print(f"Unable to update data: StockDailyPrices | Date range: from {start_date} to {end_date}."
                      f"Check your data again!")
                return

            result_dict = []
            for key in temp.keys():
                df_1 = temp[key]
                df_2 = df[key].unstack()
                result = pd.concat([df_1, df_2], join='outer')
                result.sort_index(inplace=True)
                result_dict[key] = result

            if fq:
                string = "Fq"
            else:
                string = 'Raw'
            file_path = os.path.join(output_path, f'StockDailyPrices{string}.pkl')
            with open(file_path, 'wb') as f:
                pickle.dump(result_dict, f)


class StockDailyPrices:
    def __init__(self, fq, logger=None):
        self.logger = logger
        self.data = DataTypeOperator.load_StockDailyPrices(fq)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    # start_date = '20150101'
    # end_date = '20210610'
    # DataTypeOperator.create_StockDailyPrices(start_date, end_date, fq=True)
    # DataTypeOperator.create_StockDailyPrices(start_date, end_date, fq=False)
    # DataTypeOperator.update_StockDailyPrices()
    # import time
    # time_start = time.time()
    # result_dic = DataTypeOperator.load_StockDailyPrices(True)
    # print('Time consume: %.2f seconds' % (time.time() - time_start))
    # print(result_dic['close'])
    abc = StockDailyPrices(True)
    print(abc['close'])
