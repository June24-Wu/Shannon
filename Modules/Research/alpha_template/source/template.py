import DataAPI as api
import Platform.utils.normalization as norm
import pandas as pd
import os
import datetime
import numpy as np
from Platform.factors.alpha import Alpha

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 500)


class Template(Alpha):
    update_data_count = {"DayilyPrices": 1}

    def __init__(self, alpha_name, logger=None, **kwargs):
        super().__init__(alpha_name, logger, **kwargs)

    def initializeData(self, start_time, end_time):
        num_count = Template.update_data_count["DayilyPrices"]
        while num_count > 0:
            start_time = api.get_last_trading_day(start_time)
            num_count -= 1
        self.DayilyPrices = api.get_stock_prices(ticker=None, start_date=start_time, end_date=end_time,
                                                 freq='5min', fq=True)
        df_close = self.DayilyPrices['close'].unstack()
        df_new = df_close.pct_change(fill_method='ffill')
        return df_new

    @staticmethod
    def handler(group, output):
        date_range, dataframe, alpha_name = group
        trading_date = date_range[-1]
        stock_list = api.get_universe(trading_date, universe='Float2000').index.get_level_values(1)
        dataframe = dataframe.reindex(columns=stock_list)
        tail = dataframe.tail(6).T.sum(axis=1)
        stock_list_filter = tail.loc[tail >= tail.quantile(0.5)].index
        dataframe = dataframe.reindex(columns=stock_list_filter)
        df_corr = dataframe.corr()
        df_result = df_corr.where(df_corr >= 0.7, 0)
        df_result = df_result.where(df_corr < 0.7, 1)
        df_result = df_result.sum() - 1
        df_result = df_result.to_frame(alpha_name)
        df_result[alpha_name] = np.where(df_result[alpha_name] > 0, df_result[alpha_name].max() - df_result[alpha_name],
                                         0)
        df_result = df_result.reindex(stock_list).fillna(0).sort_values(alpha_name, ascending=False)
        df_result.reset_index(inplace=True)
        df_result.insert(0, 'trading_time', pd.Timestamp(trading_date) + pd.Timedelta(hours=9, minutes=30))

        if output is not None:
            output_folder = os.path.join(output, str(trading_date.year))
            if not os.path.exists(output_folder):
                os.makedirs(output_folder, exist_ok=True)
            file_name = os.path.join(output_folder, datetime.datetime.strftime(trading_date, '%Y%m%d') + '.csv')
            df_result.to_csv(file_name, header=False, index=False)
        return df_result

    def generate_template(self, start_time, end_time, output=None, mode='dev'):
        alpha_name = "template"
        df_pct = self.initializeData(start_time, end_time)
        trading_days = api.get_trading_days(start_time, api.get_next_trading_day(end_time))
        time_sliced = [trading_days[i:i + 4] for i in range(len(trading_days) - 4)]
        grouped = [(date_range, df_pct.loc[(df_pct.index >= pd.Timestamp(date_range[0])) &
                                           (df_pct.index < pd.Timestamp(date_range[-1]))], alpha_name)
                   for date_range in time_sliced]

        if output is not None:
            output_folder = os.path.join(output, alpha_name)
        else:
            output_folder = None

        if mode == 'dev':
            import multiprocessing as mp
            import time
            from tqdm import tqdm
            workers = mp.cpu_count()
            pool = mp.Pool(processes=workers)
            stock_iter = iter(grouped)

            # 初始化任务
            result_list = [pool.apply_async(getattr(eval(alpha_name), "handler"),
                                            args=(next(stock_iter), output_folder,))
                           for _ in range(min(workers, len(grouped)))]

            flag = 1
            df_list = []
            with tqdm(total=len(grouped), ncols=150) as pbar:
                while len(result_list) > 0:
                    time.sleep(0.00001)
                    status = np.array(list(map(lambda x: x.ready(), result_list)))
                    if any(status):
                        index = np.where(status == True)[0].tolist()
                        count = 0
                        while index:
                            out_index = index.pop(0) - count
                            df = result_list[out_index].get()
                            if df is not None and not df.empty:
                                df_list.append(df)
                            result_list.pop(out_index)
                            count += 1
                            pbar.set_description(f"Calculating {alpha_name} value...")
                            pbar.update(1)
                            if flag == 1:
                                try:
                                    result_list.append(
                                        pool.apply_async(getattr(eval(alpha_name), "handler"),
                                                         args=(next(stock_iter), output_folder,)))
                                except StopIteration:
                                    flag = 0

            pool.terminate()
            return pd.concat(df_list, copy=False)

        else:
            df_list = []
            for group in grouped:
                df_list.append(getattr(eval(alpha_name), "handler")(group, output_folder))
            return pd.concat(df_list, copy=False)


if __name__ == '__main__':
    alpha = Template("template")
    output_path = '/home/liguichuan/桌面/features'
    alpha.create(start_time='20160101', end_time='20210101', output=output_path, mode='dev')
