import pandas as pd


class DataCollector:

    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path

    def read_csv_to_df(self, file_name):
        df = pd.read_csv(f'{self.data_folder_path}/{file_name}',
                         sep='\t',
                         header=None,
                         names=range(1, 168),
                         encoding="utf-8")

        return df

