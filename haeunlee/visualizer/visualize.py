import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self,target):
        self.target = target

    def read_data(self):
        self.raw_data = pd.read_csv(self.data_path, sep='\t', header=None)
        self.raw_data.columns = [str(col+1) for col in self.raw_data.columns]
        print(self.raw_data.columns)

        return self

    def str_col_visualize(self,data):
        print("string columns are ", data.columns)

        y_data = self.raw_data[self.target]
        for col in data.columns:
            x_data = data[col]


    def num_col_visualize(self,data):
        print("numerical columns are ", data.columns)



def read_data(data_path):
    raw_data = pd.read_csv(data_path, sep='\t', header=None)
    raw_data.columns = [str(col+1) for col in raw_data.columns]
    return raw_data

def main():
    data_path = 'data/lezhin_dataset_v2_training.tsv'
    raw_data = read_data(data_path)
    data_visualizer = DataVisualizer(target = '1')

    # platform - shows platform 5 holds 40 %
    platform = raw_data.iloc[:,1:5] # onehot encoding
    ratio = platform.sum().apply(lambda x: x/ len(platform))
    print("platform ratio \n", ratio)

    fig = ratio.plot(kind = 'bar', figsize=(16,12),fontsize=20,title = 'platform ratio').get_figure()
    fig.savefig('./haeunlee/visualizer/platform_ratio.png')

    print(raw_data['6'].isnull().sum())
    # data_visualizer.num_col_visualize(raw_data.select_dtypes(exclude='object'))
    # data_visualizer.str_col_visualize(raw_data.select_dtypes(include='object'))







if __name__ == "__main__":
    main()
