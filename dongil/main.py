from DataCollector.DataCollector import DataCollector


if __name__ == '__main__':
    dataCollector = DataCollector(data_folder_path='../data')
    df = dataCollector.read_csv_to_df('lezhin_dataset_v2_test.tsv')
    print(df.columns)
