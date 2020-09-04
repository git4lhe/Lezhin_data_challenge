from haeunlee.core.utils import *
import numpy as np


class ExperimentSettings(object):
    def __init__(self, data_path, target_col):
        self.data_path = data_path
        self.target_col = target_col
        self.numeric_cols = []
        self.str_cols = []
        self.ignore_cols = []

    def read_data(self, ignore=None):
        inputs = pd.read_csv(self.data_path, sep="\t", header=None, na_values=np.nan)
        inputs.columns = [str(col + 1) for col in inputs.columns.values]

        if ignore:
            inputs = inputs.drop(ignore, axis="columns")

        print(f"# Validate target column...")
        inputs = drop_y_nan_row(inputs, target_col=self.target_col)
        # inputs = inputs.drop(ignore)

        print("# Seperate X and y...")
        self.X, self.y = split_X_y(inputs, target_col=self.target_col)

        print("# classify columns to category and numerical")
        # self.numeric_cols, self.cat_cols, self.ignore = classify_cols(self.X)
        self.numeric_cols = self.X.select_dtypes(include="number").columns
        self.str_cols = self.X.select_dtypes(exclude="number").columns
        self.ignore_cols = ignore

        self.X[self.str_cols] = self.X[self.str_cols].astype(str)
        print(self.X[self.str_cols].dtypes)

