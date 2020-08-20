from haeunlee.core.utils import *


class ExperimentSettings(object):
    def __init__(self, data_path, target_col):
        self.data_path = data_path
        self.inputs = None
        self.target_col = target_col

    def read_data(self):
        inputs = pd.read_csv(self.data_path, sep="\t", header=None)
        inputs.columns = [
            str(i) for i in range(1, len(inputs.columns) + 1)
        ]  # only for lezhin data

        print(f"# Validate target column...")
        inputs = drop_nan_row(inputs, target_col=self.target_col)

        print("# Seperate X and y...")
        self.X, self.y = split_X_y(inputs, target_col=self.target_col)

        print("# classify columns to category and numerical")
        self.numeric_cols, self.cat_cols, self.ignore = classify_cols(self.X)

    def set_ignore(self, ignore_cols, show=False):
        print(ignore_cols)
        for col in ignore_cols:
            if type(col) != str:
                col = str(col)

            self.ignore.append(col)
            if col in self.numeric_cols:
                self.numeric_cols.remove(str(col))
            elif col in self.cat_cols:
                self.cat_cols.remove(str(col))
        if show:
            print("Set ignore ...")
            print(
                f"num:{sorted(self.numeric_cols)}\n"
                f"cat:{sorted(self.cat_cols)}\n"
                f"ignore:{sorted(self.ignore)}"
            )
