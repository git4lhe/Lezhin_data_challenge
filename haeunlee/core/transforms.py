import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

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
    ("impute_nan_cat", SimpleImputer(strategy="constant", fill_value=np.nan),),
    ("onehotencoder", OneHotEncoder()),
]


class PipelineCreator:
    def __init__(self, numeric_cols, category_cols, ignore):
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
        self.category_cols = category_cols

    def write_steps(self):
        pass

    def get_pipeline(self):
        final_steps = []

        print(f"Pipeline numerical({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Pipeline category({len(self.category_cols)}): {self.category_cols}")

        if self.numeric_cols:
            final_steps.append(
                (
                    "numerical transformation",
                    Pipeline(self.num_steps),
                    self.numeric_cols,
                )
            )
        if self.category_cols:
            final_steps.append(
                (
                    "categorical transformation",
                    Pipeline(self.cat_steps),
                    self.category_cols,
                )
            )

        pipe = ColumnTransformer(final_steps, remainder="drop", verbose=True)

        return pipe

    def add_num_steps(self, **step):
        for item, value in step:
            print(item, value)

    def add_cat_steps(self, **step):
        for item, value in step:
            print(item, value)
