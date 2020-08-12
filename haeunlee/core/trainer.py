from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
    cross_val_score,
)
import joblib
import os
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from os.path import join as pjoin


class ModelTrainer(object):
    def __init__(
        self, xp, preprocessor, model, pipeline_save=True, model_save=True, cv=5
    ):
        self.xp = xp
        self.preprocessor = preprocessor
        self.model = model
        self.pipeline_save = pipeline_save
        self.model_save = model_save
        self.cv = cv
        self.result_path = pjoin(self.xp.exp_dir, "2_result")

    def run_all(self):
        model, tun_par = self.model.train_settings

        # TODO: make directory for model save
        clf = Pipeline(steps=[("full_pipeline", self.preprocessor), ("model", model)])
        opt = GridSearchCV(clf, tun_par, n_jobs=-1, cv=self.cv)
        opt.fit(self.xp.inputs, self.xp.targets)

        print("\nBest parameter (CV score=%0.3f):" % opt.best_score_)

        # show results
        csv_path = pjoin(self.result_path, "train")
        # make_dir(csv_path)

        result_file = ["best_params", "cv_results"]
        results = [opt.best_params_, opt.cv_results_]
        for name, result in zip(result_file, results):
            df = pd.DataFrame(result, index=[0])
            df.to_csv(csv_path + "/" + name, index=False)

        if self.model_save:
            print("model save to.. ", self.result_path)
            joblib.dump(
                opt.best_estimator_, self.result_path + "/model_checkpoint.joblib"
            )


#
# class ModelTester(object):
#     def __init__(self, data_path, target, model_path):
#         self.data_path = data_path
#         self.target = target
#         self.model_path = model_path
#         self.report_path = os.path.dirname(self.model_path) + "/test"
#         make_dir(self.report_path)
#
#     def run(self):
#         make_dir(self.report_path)
#
#         x_data, y_data = split_target(self.data_path, self.target)
#         trained_model = joblib.load(self.model_path)
#
#         y_pred = trained_model.predict(x_data)
#         write_report(y_pred,y_data,save_path=self.report_path)
