import warnings
import pandas as pd
from sklearn.base import BaseEstimator


class PurchaseRatioMaker(BaseEstimator):

    def __init__(self, y_col, count_threshold=100, new_col_prefix='purchase_ratio'):
        self.y_col = y_col
        self.count_threshold = count_threshold
        self.new_col_prefix = new_col_prefix

        self.mean_df_dict = dict()
        self.y_mean = 0

    def fit(self, X, y=None):
        df = pd.concat([X, y], axis=1)
        self.y_mean = df[self.y_col].mean()

        for x_col in df.columns:
            if x_col == self.y_col:
                continue

            mean_df = df.groupby(x_col)[self.y_col].mean().rename('_'.join([self.new_col_prefix, x_col]))
            count_df = df.groupby(x_col)[self.y_col].count()
            over_threshold_idx = count_df[count_df > self.count_threshold].index
            self.mean_df_dict[x_col] = mean_df[over_threshold_idx]

    def transform(self, X):
        for x_col, mean_df in self.mean_df_dict.items():
            X = pd.merge(X, mean_df, how='left', left_on=x_col, right_index=True)
            X['_'.join([self.new_col_prefix, x_col])].fillna(self.y_mean, inplace=True)

        return X

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
