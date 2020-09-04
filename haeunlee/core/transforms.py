import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

join = FunctionTransformer(' '.join, validate=True)
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
num_steps = [
    (
        "impute_nan_num",
        SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=0, add_indicator=True
        ),
    ),
    ("standardscaler", StandardScaler()),
]

cat_steps = [
    (
        "impute_nan_cat",
        SimpleImputer(
            missing_values=np.nan, strategy='constant', fill_value='ABCD1234', add_indicator=True
        ),
    ),
    # ("join", join),
    ("HashingVectorizer", HashingVectorizer(n_features=2 ** 5, binary = False, lowercase=False)),
]


class PipelineCreator:
    def __init__(self, numeric_cols, str_cols, ignore = None):
        """
        { 
           imputation: nan/unknown -> 같은 데이터로 처리, another category
           standard scaler: 
           onehotencoder:
        }
        """
        self.num_steps = num_steps
        self.cat_steps = cat_steps
        self.numeric_cols = numeric_cols
        self.str_cols = str_cols
        self.final_pipe = []

    def get_pipeline(self):

        print(f"Pipeline numerical({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Pipeline string({len(self.str_cols)}): {self.str_cols}")

        self.final_pipe.append(
            ("numerical", Pipeline(self.num_steps), self.numeric_cols)
        )
        self.final_pipe.append(
            ("string column transformation", Pipeline(self.cat_steps), self.str_cols)
        )
        pipe = ColumnTransformer(self.final_pipe, remainder="drop", verbose=True)

        return pipe

    def add_pipeline(self, **step):
        print(step)
