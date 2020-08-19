import warnings
import pandas as pd
from sklearn.base import BaseEstimator


class DataCleanser(BaseEstimator):

    def __init__(self, na_threshold=0.1, cate_threshold=1):
        self.na_threshold = na_threshold
        self.cate_threshold = cate_threshold

        self.na_col_list = list()
        self.str_col_list = list()
        self.cate_col_list = list()
        self.drop_col_list = list()

    def _make_na_col_list(self, df):
        self.na_col_list = df.columns[df.isna().sum() >= self.na_threshold * len(df)].tolist()

    def _make_str_col_list(self, df):
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                self.str_col_list.append(col)

    def _make_cate_col_list(self, df):
        self.cate_col_list = df.columns[df.nunique() <= self.cate_threshold].tolist()

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        self._make_na_col_list(df)
        self._make_str_col_list(df)
        self._make_cate_col_list(df)

        self.drop_col_list = list(set(self.na_col_list + self.str_col_list + self.cate_col_list))

    def transform(self, X):
        df = pd.DataFrame(X)
        remain_cols = [x for x in df.columns if x not in self.drop_col_list]
        return df[remain_cols].to_numpy()

    def fit_transform(self, X, y=None):
        self.fit(X=X, y=y)
        return self.transform(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn('From version 0.24, get_params will raise an '
                              'AttributeError if a parameter cannot be '
                              'retrieved as an instance attribute. Previously '
                              'it would return None.',
                              FutureWarning)
                value = None
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out
