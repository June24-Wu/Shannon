import pandas as pd


def check(key, end_date, df: pd.DataFrame):
    cols = ['alpha_value', 'symbol', 'trading_time']

    try:
        df = df.rename(columns={key: 'alpha_value', 'timestamp': 'trading_time'})
        df_cols = sorted(df.columns.tolist())

        if df_cols != cols:
            print("columns error")
            return

        df = df.sort_values(by="trading_time", ascending=False)
        df['trading_time'] = df['trading_time'].dt.strftime('%Y-%m-%d')

        trading_time_list = df['trading_time'].unique()
        if len(trading_time_list) > 1:
            print("date error.")
            return
        last_date = df['trading_time'].iloc[0]
        if end_date != last_date:
            print("date error.")
            return

        return df

    except Exception as e:
        print(e)
        return pd.DataFrame()
