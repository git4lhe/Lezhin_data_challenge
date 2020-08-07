from DataCollector.DataCollector import DataCollector
from DataCleanser.DataCleanser import DataCleanser
from DataCleanser.Scaler import Scaler


if __name__ == '__main__':
    dataCollector = DataCollector(data_folder_path='../data')
    df = dataCollector.read_csv_to_df('lezhin_dataset_v2_test.tsv')

    y_col = 1
    dataCleanser = DataCleanser(y_col=y_col)
    df = dataCleanser.remove_na_col(df, threshold=0.1)
    df = dataCleanser.remove_str_col(df)
    df = dataCleanser.remove_cate_col(df, threshold=2)

    x_cols = [x for x in df.columns if x != y_col]
    scaler = Scaler(y_col=y_col, x_cols=x_cols, scaler_method='MinMax')
    scaler.fit_scaler(df)
    df = scaler.transform_df(df)

    print(df)

