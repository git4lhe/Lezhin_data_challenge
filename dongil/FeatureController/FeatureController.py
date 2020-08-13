import pandas as pd
from sklearn.impute import SimpleImputer

class FeatureController:

    def __init__(self, y_col):
        self.y_col = y_col

        self.fill_value = 2.0
        self.count_threshold = 100
        self.new_col_name = 'purchase_ratio'

    def custom_imputation(self, df, impute_col_list):
        imp = SimpleImputer(strategy='constant', fill_value=self.fill_value)
        df[impute_col_list] = imp.fit_transform(df[impute_col_list].to_numpy())

    def make_purchase_ratio_column(self, df, x_col):
        y_mean = df[self.y_col].mean()

        mean_df = df.groupby(x_col)[self.y_col].mean().rename(self.new_col_name)
        count_df = df.groupby(x_col)[self.y_col].count()
        over_threshold_idx = count_df[count_df > self.count_threshold].index

        merge_df = pd.merge(df, mean_df[over_threshold_idx],
                            how='left', left_on=x_col, right_index=True)
        merge_df[self.new_col_name].fillna(y_mean, inplace=True)

        return merge_df
