import os
import shutil
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import pandas as pd

from sklearn.model_selection import train_test_split

def drop_nan_row(df ,target_col):
    return df.dropna(subset=[target_col])

def drop_nan_col(df,threshold=0.8):
    return

def split_X_y(df,target_col):
    # input = df.drop(target_col, axis=1)
    # target = df[target_col]
    return df.drop(target_col, axis=1), df[target_col]

def classify_cols(df,drop_ratio = 0.7,unique_value=100,show=False):
    category, numerical, ignore = [], [], []
    length = len(df.index)

    for col in df.columns:
        ratio_of_null = df[col].isnull().sum() / length
        if drop_ratio < ratio_of_null:
            ignore.append(col)

        else:
            # categorical
            if df[col].dtype == object:
                if len(df[col].value_counts()) > unique_value:
                    ignore.append(col)
                else:
                    category.append(col)
            # numerical
            else:
                numerical.append(col)
    if show:
        print(f"numerical({len(numerical)}): {numerical}")
        print(f"category({len(category)}): {category}")
        print(f"ignore({len(ignore)}): {ignore}")


    return numerical, category, ignore