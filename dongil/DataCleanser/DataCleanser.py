import pandas as pd


class DataCleanser:

    def __init__(self, y_col):
        self.y_col = y_col

        # for debugging
        self.na_col_list = list()
        self.str_col_list = list()
        self.cate_col_list = list()

    def remove_na_col(self, df, threshold):
        self.na_col_list = df.columns[df.isna().sum() >= threshold * len(df)].tolist()

        not_na_col_list = [x for x in df.columns if x not in self.na_col_list]
        not_na_col_list.append(self.y_col)
        not_na_col_list_with_y = list(set(not_na_col_list))

        return df[not_na_col_list_with_y]

    def remove_str_col(self, df):
        not_str_col_list = list()
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
                not_str_col_list.append(col)
            except ValueError:
                self.str_col_list.append(col)

        not_str_col_list.append(self.y_col)
        not_str_col_list_with_y = list(set(not_str_col_list))

        return df[not_str_col_list_with_y]

    def remove_cate_col(self, df, threshold):
        self.cate_col_list = df.columns[df.nunique() <= threshold].tolist()

        not_cate_col_list = [x for x in df.columns if x not in self.cate_col_list]
        not_cate_col_list.append(self.y_col)
        not_cate_col_list_with_y = list(set(not_cate_col_list))

        return df[not_cate_col_list_with_y]

