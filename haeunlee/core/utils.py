import os
import shutil
import pandas as pd
from os.path import join as pjoin
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import pandas as pd
from sklearn.model_selection import train_test_split


class ExperimentSettings(object):
    def __init__(self, arguments):
        self.exp_dir = pjoin("./Project_Template", str(datetime.now()).strip())
        self.data_files = arguments.train
        self.test_files = arguments.test
        self.target = arguments.target
        self.data_name = self.data_files.split("/")[-1]
        self.classifier = ""
        self.optimization = False
        self.split_ratio = 0.2
        self.report_dir = "report"
        self.report_csv = arguments.report
        self.predict_csv = arguments.predict
        self.model_save = "model_checkpoint"

        self.data_path = ""
        self.split_data_path = ""
        self.numeric_features = []
        self.categorical_features = []

    def create_setting(self, split=False):
        self.add_raw_source()
        self.summarize_data()
        if split:
            self.add_split_data()

    def add_raw_source(self):
        raw_data_folder = pjoin(self.exp_dir, "0_data", "0_raw")
        make_dir(raw_data_folder)
        data_folder = pjoin(self.exp_dir, "0_data", "1_validated")
        make_dir(data_folder)

        # delte samples with nan values in target
        self.data_path = pjoin(data_folder, self.data_name)

        raw_data = load_data(
            input_path=self.data_files,
            target_name=self.target,
            save=True,
            save_path=pjoin(raw_data_folder, self.data_name),
        )

        categorical_features = []
        numeric_features = []

        for col, d_type in zip(raw_data.columns, raw_data.dtypes):
            if col == self.target:
                continue
            if d_type == "object":
                categorical_features.append(col)
            else:
                numeric_features.append(col)

        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        raw_data.to_csv(self.data_path, index=False)
        print("Created data csv file at", self.data_path)

    def summarize_data(self):
        data_folder = pjoin(self.exp_dir, "1_summary")
        make_dir(data_folder)

        # visualzation of raw data
        raw_data = pd.read_csv(self.data_path)
        for cols, d_type in zip(raw_data.columns, raw_data.dtypes):
            if d_type == "object":
                raw_data[cols].value_counts().plot(kind="barh")
                plt.title("{}\n[discrete data]".format(cols))
                plt.savefig("{}/{}.png".format(data_folder, cols))
            else:
                raw_data[cols].hist()
                plt.title("{}\n[continuous data]".format(cols))
                plt.savefig("{}/{}.png".format(data_folder, cols))
            plt.close()

        print("Created data summary at", data_folder)

    # split data -> preprocessed data does not have to be saved
    def add_split_data(self):
        self.split_data_path = pjoin(self.exp_dir, "0_data", "2_split")
        make_dir(self.split_data_path)

        raw_data = pd.read_csv(self.data_path)
        train, test = train_test_split(raw_data, test_size=self.split_ratio)

        train.to_csv(
            self.split_data_path + "/train_{}".format(self.data_name), index=False
        )
        test.to_csv(
            self.split_data_path + "/val_{}".format(self.data_name), index=False
        )

        print("Created data csv file at", self.split_data_path)


def make_dir(path, previous=False):
    """
    previous: delete all previous files
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def check_source(path):
    if not os.path.exists(path):
        return None

    return True


def load_data(input_path, target_name, target_path=None, save=False, save_path=None):
    raw_data = pd.read_csv(input_path)
    if save:
        make_dir(os.path.dirname(save_path))
        raw_data.to_csv(save_path)
        print("Created data csv file at", save_path)

    print("Data size before empty lines removal:", len(raw_data))
    if not target_name in raw_data:
        raw_y_data = pd.read_csv(target_path)
        assert len(raw_data) == len(raw_y_data)
        raw_data.merge(raw_y_data)

    index_num = raw_data[raw_data[target_name].isnull()].index
    print("Data size after  empty lines removal:", len(raw_data) - len(index_num))

    return raw_data.drop(index_num)


def split_target(input_path, target):
    data = pd.read_csv(input_path)

    y_data = data[target]
    x_data = data.drop(target, axis=1)

    return x_data, y_data
