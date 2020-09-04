from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from haeunlee.core.utils import make_dir
from sklearn.linear_model import Lasso
from dask.distributed import Client, progress
from sklearn.svm import SVC
import joblib
import time
from time import gmtime, strftime
from sklearn.ensemble import RandomForestClassifier
import dask_ml.model_selection as dcv

client = Client(processes=False, threads_per_worker=8, n_workers=1, memory_limit="16GB")


class ParameterGrid:
    @property
    def random_forest(self):
        param_grid = {
            'clf__n_estimators': [100, 300, 500],
        }
        # max_depth = [5, 8, 15, 25, 30]
        # min_samples_split = [2, 5, 10, 15, 100]
        # min_samples_leaf = [1, 2, 5, 10]

        return param_grid

    @property
    def svm(self):
        C = [0.001, 0.01]
        kernel = ["sigmoid"]

        param_grid = dict(C=C, kernel=kernel)

        return param_grid

    @property
    def lasso(self):
        param_grid = {"alpha": [0.02, 0.024]}  # , 0.025, 0.026, 0.03]}

        return param_grid


class ModelTrainer(object):
    def __init__(self, preprocessor, pipeline_save=True, model_save=True, cv=5):
        self.preprocessor = preprocessor
        self.pipeline_save = pipeline_save
        self.model_save = model_save
        self.cv = cv
        self.result_path = time.ctime()
        self.param_grid = ParameterGrid()

    def dump(self, grid_search, model):
        """
        The training time is written in the file
        """
        results = pd.DataFrame(grid_search.cv_results_.head()).head()
        print(grid_search.cv_results_)
        print(grid_search.best_estimator_)

        make_dir(f"haeunlee/result/{model}")
        results.to_csv(
            f"haeunlee/result/{model}/{strftime('%Y-%m-%d %H:%M:%S', gmtime())}.csv"
        )

    def run_rf(self,data_x, data_y):
        clf = RandomForestClassifier(
            n_jobs=2, max_features="sqrt", n_estimators=50, oob_score=True
        )
        estimator = Pipeline(steps=[('preprocessor',self.preprocessor), ('clf', clf)])

        grid_rf = GridSearchCV(estimator=estimator, param_grid=self.param_grid.random_forest, cv=self.cv,verbose=3)
        grid_rf.fit(data_x, data_y)
        print("The best score of random forest", grid_rf.best_score_)


# TODO
"""
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
"""
