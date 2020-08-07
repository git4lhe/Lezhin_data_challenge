import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


class Scaler:

    def __init__(self, y_col, x_cols, scaler_method='Standard'):
        self.y_col = y_col
        self.x_cols = x_cols

        self.scaler_name = 'scaler.sav'

        if scaler_method == 'Standard':
            self.scaler = StandardScaler()
        elif scaler_method == 'MinMax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = RobustScaler()

    def fit_scaler(self, df):
        df_x = df[self.x_cols]
        self.scaler.fit_transform(df_x.to_numpy())

    def save_scaler(self):
        joblib.dump(self.scaler, self.scaler_name)

    def load_scaler(self):
        self.scaler = joblib.load(self.scaler_name)

    def transform_df(self, df):
        df_x = df[self.x_cols].copy()

        df_x = self.scaler.transform(df_x.to_numpy())
        df_x = pd.DataFrame(data=df_x, columns=self.x_cols)
        df[self.x_cols] = df_x

        return df
