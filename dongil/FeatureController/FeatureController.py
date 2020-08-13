import pandas as pd
from sklearn.impute import SimpleImputer

class FeatureController:

    def __init__(self, y_col):
        self.y_col = y_col

        self.fill_value = 2.0
        self.count_threshold = 100
        self.right_suffix = '_mean'

    def custom_imputation(self, df, impute_col_list):
        imp = SimpleImputer(strategy='constant', fill_value=self.fill_value)
        df[impute_col_list] = imp.fit_transform(df[impute_col_list].to_numpy())

    def make_purchase_ratio_column(self, df, x_col):
        y_mean = df[self.y_col].mean()

        mean_df = df.groupby(x_col)[self.y_col].mean()
        count_df = df.groupby(x_col)[self.y_col].count()
        over_threshold_idx = count_df[count_df > self.count_threshold].index

        merge_df = pd.merge(df, mean_df[over_threshold_idx],
                            how='left', left_on=x_col, right_index=True,
                            suffixes=('', self.right_suffix))
        merge_df[''.join([str(self.y_col), self.right_suffix])].fillna(y_mean, inplace=True)

        return merge_df
