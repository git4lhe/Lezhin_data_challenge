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

class PipelineCreator:
    def __init__(self):
        """
        { 
           imputation: nan/unknown -> 같은 데이터로 처리, another category
           standard scaler: 
           onehotencoder:
        }
        """
        pass

class NumPipelineCreator():

    def __init__(self):
        # basic numerical pipeline
        self.steps = [
            ("impute_nan_num",SimpleImputer(
                    missing_values=np.nan, strategy="median", add_indicator=True)
             ),
            ("standardscaler", StandardScaler())
            ]

    def add_transform(self, steps):
        self.steps.append((steps))
        for step in steps:
            print(step)

    def get_pipeline(self):
        numeric_transformer = Pipeline(steps=self.steps)
        return numeric_transformer

class CatPipelineCreator:
    def get_pipeline(self):
        # basic numerical pipeline
        categorical_transformer = Pipeline(
            steps=[
                (
                    "impute_nan_cat",
                    SimpleImputer(strategy="constant", fill_value="unknown"),
                ),
                ("onehotencoder", OneHotEncoder()),
            ]
        )
        return categorical_transformer

