# import os
# import sys
# sys.path.append(os.path.realpath('.'))
# from main import *
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from core.utils import make_dir
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
    OneHotEncoder,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator


class DataTransformer(object):
    def __init__(self, input_path, target_name, target_path=None, ratio=0.3):
        self.input_path = input_path
        self.target_name = target_name
        self.target_path = target_path
        self.ratio = ratio

    def split_target(self):
        """
        pf : whole data
        """
        print(self.input_path)
        raw_data = pd.read_csv(self.input_path)

        # target as another csv file
        if self.target_path:
            raw_y_data = pd.read_csv(self.target_path)
            assert len(raw_data) == len(raw_y_data)
            raw_data = pd.merge(raw_data, raw_y_data, how="outer")

        # target in data_path
        assert self.target_name in raw_data.columns

        # delete row with y as null
        index_num = raw_data[raw_data[self.target_name].isnull()].index
        self.pf = raw_data.drop(index_num)
        self.target = self.pf[self.target_name]
        self.data = self.pf.drop(self.target_name, axis=1)

        return raw_data

    def split_train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=self.ratio, shuffle=False
        )

        print("# OF TRAIN DATASET:", len(self.x_train))
        print("# OF TEST DATASET:", len(self.x_test))

        return self.x_train, self.x_test, self.y_train, self.y_test

    def visualization(self):
        folder_save = "./visualization"
        make_dir(folder_save)

        for cols, d_type in zip(self.pf.columns, self.pf.dtypes):
            f = self.pf[cols].unique()
            if d_type == "object":
                self.pf[cols].value_counts().plot(kind="barh")
                plt.title("{}\n[discrete data]".format(cols))
                plt.savefig("{}/{}.png".format(folder_save, cols))
            else:
                self.pf[cols].hist()
                plt.title("{}\n[continuous data]".format(cols))
                plt.savefig("{}/{}.png".format(folder_save, cols))

            plt.close()

        return self


class PipelineCreator:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        print("numeric_features:\n", self.numeric_features)
        print("categorical_features:\n", self.categorical_features)

    def make_pipeline(self):
        """
        { 
           imputation: nan/unknown -> 같은 데이터로 처리, another category
           standard scaler: 
           onehotencoder:
        }
        """
        # self.seperate_cols(data,target)
        categorical_transformer = Pipeline(
            steps=[
                (
                    "impute_nan_cat",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),
                ("onehotencoder", OneHotEncoder()),
            ]
        )
        numeric_transformer = Pipeline(
            steps=[
                (
                    "impute_nan_num",
                    SimpleImputer(
                        missing_values=np.nan, strategy="median", add_indicator=True
                    ),
                ),
                ("standardscaler", StandardScaler()),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numeric_transformer, self.numeric_features),
                ("categorical", categorical_transformer, self.categorical_features),
            ],
            remainder="drop",
            verbose=True,
        )

        return preprocessor
