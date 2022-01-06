import datetime


def convert_to_standard_daily_data_csv(df: pd.DataFrame, output_name: str, output_path: str):
    grouped = df.groupby('timestamp')
    for date, group in grouped:
        date_format = pd.to_datetime(date).date()
        assert DataAPI.is_trading_day(date), f"{date} is not a trading date!"
        file_name = datetime.date.strftime(date_format, '%Y%m%d') + '.par'
        folder = os.path.join(output_path, output_name, str(date_format.year))
        if not os.path.exists(folder):
            os.makedirs(folder)
        file = os.path.join(folder, file_name)
        group.to_parquet(file)


class DataLoader(object):
    def __init__(self, model, device=None):
        self.feature_data = None
        self.target = None
        self.length = None
        self.y = None

    def load_data_from_file(data_path, end_date, start_date="2015-01-01"):
        time_list = DataAPI.get_trading_days(start_date, end_date)
        final = []
        for date in time_list:
            date = data_path + str(date.year) + '/' + str(date.year) + str(date.month).zfill(2) + str(date.day).zfill(
                2) + ".par"
            if (os.path.exists(date)):
                final.append(pd.read_parquet(date))
        self.feature_data = pd.concat(final, axis=0)
        self.length = self.feature_data.shape[0]
        return self.feature_data

    def to_torch_DataLoader(sequence, shuffle, batch_size=1024, num_workers=16):
        self.target = pd.DataFrame(self.feature_data["target"])
        self.feature_data.drop("target", axis=1, inplace=True)
        self.feture_data.set_index(["timestamp", "ticker"], inplace=True)
        x = torch.from_numpy(np.array(self.feture_data).reshape(self.length, -1, sequence))
        y = torch.from_numpy(np.array(self.target).reshape(-1, 1))
        self.feature_data = torch.utils.data.TensorDataset(x, y)
        self.feature_data = torch.utils.data.DataLoader(
            dataset=self.feature_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        return self.feature_data


