import unittest
import argparse
import pandas as pd
from sklearn.datasets import load_iris
from random import random
import numpy as np
from haeunlee.core.transforms import *


class TestSummary(unittest.TestCase):
    def test_NumPipelineCreator_imputer(self):
        # creates new column if there was np.nan column
        raw_data = {
            "age": [np.nan, 52, 36, 24, 73],
            "preTestScore": [4, 24, 31, 2, 3],
            "postTestScore": [25, 94, 57, 62, 70],
        }
        df = pd.DataFrame(raw_data, columns=["age", "preTestScore", "postTestScore"])

        pipeline = NumPipelineCreator()
        full_pipeline = ColumnTransformer(
            [("num_transform", pipeline.get_pipeline(), df.columns)]
        )
        result = full_pipeline.fit_transform(df)

        print("\n", result)

    def test_CatPipelineCreator_onehot(self):
        # creates new column if there was np.nan column
        raw_data = {
            "name": ["Dave", "Joe", "Julia"],
            "grade": ["A", "B", "C"],
            "gender": ["M", "M", "F"],
        }
        df = pd.DataFrame(raw_data, columns=["name", "grade", "gender"])

        pipeline = CatPipelineCreator()
        full_pipeline = ColumnTransformer(
            [("num_transform", pipeline.get_pipeline(), df.columns)]
        )

        result = full_pipeline.fit_transform(df)
        print("\n", result)

    def test_ColumnTransformer(self):
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        df = pd.DataFrame(
            {"city": ["London", "London", "Paris", "Sallisaw"], "rating": [5, 3, 4, 5]}
        )
        num_pipeline = NumPipelineCreator()
        cat_pipeline = CatPipelineCreator()

        full_pipeline = ColumnTransformer(
            [
                ("num_transform", num_pipeline.get_pipeline(), ["rating"]),
                ("cat_transform", cat_pipeline.get_pipeline(), ["city"]),
            ]
        )

        result = full_pipeline.fit_transform(df)
        print("\n", result)

    def test_ColumnTransformer_drop(self):
        # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html
        df = pd.DataFrame(
            {
                "city": ["London", "London", "Paris", "Sallisaw"],
                "rating": [5, 3, 4, 5],
                "year": [20, 18, 21, 19],
            }
        )
        num_pipeline = NumPipelineCreator()
        cat_pipeline = CatPipelineCreator()

        full_pipeline = ColumnTransformer(
            [
                ("num_transform", num_pipeline.get_pipeline(), ["year"]),
                ("cat_transform", cat_pipeline.get_pipeline(), ["city"]),
            ],
            remainder="drop",
        )
        result = full_pipeline.fit_transform(df)
        print("\n", result)


if __name__ == "__main__":
    unittest.main()
