import pandas as pd
import DataAPI
import datetime
import os



def time_series_norm(data: pd.DataFrame,rolling:int,output_path: str):
    """
    rolling = days ex. rolling = 30
    """
    for column in data.columns:
        one_data = pd.DataFrame(data[column]).unstack()
        mean = one_data.rolling(rolling).mean()
        std = one_data.rolling(rolling).std()
        one_data = ((one_data- mean) /std ).stack()
        one_data.columns = one_data.columns + "_time_series_norm_" + str(rolling)
        __convert_to_standard_daily_data_csv(one_data, one_data.columns.to_list()[0], output_path)




def __convert_to_standard_daily_data_csv(df: pd.DataFrame, output_name: str, output_path: str):
    grouped = df.groupby('timestamp')
    for date, group in grouped:
        date_format = pd.to_datetime(date).date()
        assert DataAPI.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.csv'
        folder = os.path.join(output_path, output_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, file_name)
        group.to_csv(file,header=False, encoding='utf-8')
    return None