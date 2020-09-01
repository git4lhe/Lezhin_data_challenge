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

class ParameterGrid():

    @property
    def random_forest(self):
        n_estimators = [100, 300, 500, 800, 1200]
        # max_depth = [5, 8, 15, 25, 30]
        # min_samples_split = [2, 5, 10, 15, 100]
        # min_samples_leaf = [1, 2, 5, 10]

        param_grid = dict(
            n_estimators=n_estimators)
            # max_depth=max_depth,
            # min_samples_split=min_samples_split,
            # min_samples_leaf=min_samples_leaf,
        # )
        return param_grid

    @property
    def svm(self):
        C =[0.001, 0.01]
        kernel = ["sigmoid"]

        param_grid = dict(
            C=C,
            kernel = kernel)

        return param_grid

    @property
    def lasso(self):
        param_grid = {'alpha': [0.02, 0.024]} #, 0.025, 0.026, 0.03]}

        return param_grid


class ModelTrainer(object):
    def __init__(self, xp, xt, preprocessor, pipeline_save=True, model_save=True, cv=5):
        self.xp = xp
        self.X_t = xt
        self.preprocessor = preprocessor
        self.pipeline_save = pipeline_save
        self.model_save = model_save
        self.cv = cv
        self.result_path = time.ctime()
        self.param_grid = ParameterGrid()

    def dump(self,grid_search, model):
        '''
        The training time is written in the file
        '''
        results = pd.DataFrame(grid_search.cv_results_).head()

        make_dir(f"haeunlee/result/{model}")
        results.to_csv(f"haeunlee/result/{model}/{strftime('%Y-%m-%d %H:%M:%S', gmtime())}.csv")

    def run_dask_rf(self):
        model = "random_forest"

        clf = RandomForestClassifier()
        grid_search = GridSearchCV(
            estimator=clf, param_grid=self.param_grid.random_forest, cv=self.cv, verbose=1
        )

        with joblib.parallel_backend("dask"):
            progress(grid_search.fit(self.X_t, self.xp.y))

        self.dump(grid_search,model=model)

    def run_dask_lasso(self):
        model = 'lasso'
        grid_search = GridSearchCV(
            Lasso(), param_grid=self.param_grid.lasso, cv=self.cv
        )
        with joblib.parallel_backend("dask"):
            progress(grid_search.fit(self.X_t, self.xp.y))

        self.dump(grid_search, model=model)

    def run_dask_svc(self):
        model = 'svm'
        grid_search = GridSearchCV(
            SVC(gamma="auto", random_state=0, probability=True),
            param_grid=self.param_grid.svm, cv = self.cv
        )

        with joblib.parallel_backend("dask"):
            progress(grid_search.fit(self.X_t, self.xp.y))

        self.dump(grid_search,model=model)

    def run_all_rf(self):
        # TODO: make directory for model save
        transform_X = self.preprocessor.fit_transform(self.xp.X)
        # pipe = Pipeline([("preprocessing", self.preprocessor),
        #                  ("classifier",RandomForestClassifier\
        #                      (n_jobs=-1,max_features= 'sqrt' ,\
        #                       n_estimators=50, oob_score = True) )])
        # param_grid = {
        #                 'classifier__n_estimators': [200, 500, 700],
        #                 'classifier__max_features': ['auto', 'sqrt']
        #             }
        clf = RandomForestClassifier(
            n_jobs=-1, max_features="sqrt", n_estimators=50, oob_score=True
        )
        param_grid = {"n_estimators": [200, 500, 700], "max_features": ["auto", "sqrt"]}
        CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        CV_rfc.fit(transform_X, self.xp.y)

        print(CV_rfc.best_estimator_)
        print("The best score of random forest", CV_rfc.best_score_)

    def run_all_ridge(self):
        transform_X = self.preprocessor.fit_transform(self.xp.X)

        pipe = Pipeline([("preprocessing", self.preprocessor), ("classifier", Ridge())])
        param_grid = {
            "classifier__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        }
        CV_ridge = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3)
        CV_ridge.fit(self.xp.X, self.xp.y)

        print(CV_ridge.best_estimator_)
        print("The best score of ridge regression", CV_ridge.best_score_)

    def run_all_xgboost(self):

        pipe = Pipeline(
            [
                ("preprocessing", self.preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        learning_rate=0.02,
                        n_estimators=600,
                        objective="binary:logistic",
                    ),
                ),
            ]
        )

        param_grid = {
            "classifier__min_child_weight": [1, 5, 10],
            "classifier__gamma": [0.5, 1, 1.5, 2, 5],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__max_depth": [3, 4, 5],
        }
        CV_xgb = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3)
        CV_xgb.fit(self.xp.X, self.xp.y)

        print(CV_xgb.best_estimator_)
        print("The best score of xgboost", CV_xgb.best_score_)

    # TODO
    def run_all_dask(self):
        param_grid = {"C": [0.001], "kernel": ["rbf", "poly", "sigmoid"]}

        grid_search = GridSearchCV(
            SVC(gamma="auto", random_state=0, probability=True),
            param_grid=param_grid,
            iid=True,
            cv=3,
            n_jobs=-1,
        )

        grid_search.fit(transform_X[:10000], self.xp.y[:10000])

        print(grid_search.predict(transform_X)[:5])
        print(grid_search.score(transform_X[:10000], self.xp.y[:10000]))

    # TODO
    def result_save(self):
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
