from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
from sklearn.svm import SVC
import joblib
import os
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from sklearn.linear_model import LinearRegression
from os.path import join as pjoin


class TrainSettings():
    def hyperparameter(self):
        pass
