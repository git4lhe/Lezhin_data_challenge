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
from core.utils import split_target, make_dir, load_data
from core.report import write_report
from os.path import join as pjoin


class ModelTrainer(object):
    def __init__(
        self, xps, preprocessor, model, pipeline_save=True, model_save=True, cv=5
    ):
        self.xps = xps
        self.preprocessor = preprocessor
        self.model = model
        self.pipeline_save = pipeline_save
        self.model_save = model_save
        self.cv = cv
        self.result_path = pjoin(self.xps.exp_dir, "2_result")
        make_dir(self.result_path)

    def train_settings(self, model):
        model_dict = {"SVM": SVC(), "linear_regression": LinearRegression(), "PCA": ""}
        tuned_parameters = {
            "SVM": [
                {
                    "model__kernel": ["linear", "rbf", "poly"],
                    "model__C": [0.1, 1],
                    "model__gamma": [1e-3, 1e-4, "auto", "scale"],
                }
            ]
        }

        return model_dict[model], tuned_parameters[model]

    def run_all(self):
        model, tun_par = self.train_settings(self.model)

        # get_data
        if self.xps.split_data_path:
            print(self.xps.split_data_path)
            # TRAIN, VALIDATION

        else:
            # TRAIN
            x_data, y_data = split_target(self.xps.data_path, target=self.xps.target)
            clf = Pipeline(
                steps=[("full_pipeline", self.preprocessor), ("model", model)]
            )

            parameters = {"model__kernel": ["linear"], "model__C": [0.1]}
            opt = GridSearchCV(clf, parameters, n_jobs=-1, cv=self.cv)
            opt.fit(x_data, y_data)

            print("\nBest parameter (CV score=%0.3f):" % opt.best_score_)

            # show results
            csv_path = pjoin(self.result_path, "train")
            make_dir(csv_path)

            result_file = ["best_params", "cv_results"]
            results = [opt.best_params_, opt.cv_results_]
            for name, result in zip(result_file, results):
                df = pd.DataFrame(result, index=[0])
                df.to_csv(csv_path + "/" + name, index=False)

            if self.model_save:
                print("model save to.. ", self.result_path)
                joblib.dump(opt.best_estimator_, self.result_path + "/model_checkpoint.joblib")

class ModelTester(object):
    def __init__(self, data_path, target, model_path):
        self.data_path = data_path
        self.target = target
        self.model_path = model_path
        self.report_path = os.path.dirname(self.model_path) + "/test"
        make_dir(self.report_path)

    def run(self):
        make_dir(self.report_path)

        x_data, y_data = split_target(self.data_path, self.target)
        trained_model = joblib.load(self.model_path)

        y_pred = trained_model.predict(x_data)
        write_report(y_pred,y_data,save_path=self.report_path)



