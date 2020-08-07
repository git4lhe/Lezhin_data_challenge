from DataCollector.DataCollector import DataCollector
from DataCleanser.DataCleanser import DataCleanser


if __name__ == '__main__':
    dataCollector = DataCollector(data_folder_path='../data')
    df = dataCollector.read_csv_to_df('lezhin_dataset_v2_test.tsv')

    y_col = 1
    dataCleanser = DataCleanser(y_col=y_col)
    df = dataCleanser.remove_na_col(df, threshold=0.1)
    df = dataCleanser.remove_str_col(df)
    df = dataCleanser.remove_cate_col(df, threshold=2)

    print(dataCleanser.na_col_list)
    print(dataCleanser.str_col_list)
    print(dataCleanser.cate_col_list)
    print(df.columns)
