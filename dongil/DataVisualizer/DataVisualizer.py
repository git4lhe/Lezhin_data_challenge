import os
import pandas as pd
import matplotlib.pyplot as plt


class DataVisualizer:

    def __init__(self, y_col, save_path):
        self.y_col = y_col
        self.save_path = save_path

        self.fill_na_val = 2.0
        os.makedirs(save_path, exist_ok=True)

    def na_col_plot(self, _df, x_col):
        df = _df.copy()

        bins = [0.1 * x for x in range(12)]
        labels = [0.1 * x for x in range(10)] + [self.fill_na_val]

        df['binned'] = pd.cut(df[x_col], bins=bins, labels=labels)
        df['binned'].fillna(value=self.fill_na_val, inplace=True)

        mean_df = df.groupby('binned')[self.y_col].mean()
        count_df = df.groupby('binned')[self.y_col].count()
        names = [str(round(x, 2)) for x in mean_df.index]
        mean_values = [round(x, 3) for x in mean_df.values]
        count_ratio_values = [round(x/len(df), 3) for x in count_df.values]

        title = f'Column {x_col} Binning'
        plt.figure(figsize=(18, 6))
        plt.subplot(131)
        plt.bar(names, mean_values)
        plt.subplot(133)
        plt.bar(names, count_ratio_values)
        plt.suptitle(title)
        plt.savefig(os.path.join(self.save_path, f"{title.replace(' ', '_')}.png"), dpi=1000)

    def str_col_plot(self, _df, x_col):
        df = _df.copy()

        mean_df = df.groupby(x_col)[self.y_col].mean()
        count_df = df.groupby(x_col)[self.y_col].count()
        mean_values = [round(x, 3) for x in mean_df.values]

        title = f'Column {x_col} Grouping'
        plt.figure(figsize=(18, 6))
        plt.subplot(121)
        plt.plot(sorted(mean_values))
        plt.subplot(133)
        plt.plot(sorted(count_df.values))
        plt.suptitle(title)
        plt.savefig(os.path.join(self.save_path, f"{title.replace(' ', '_')}.png"), dpi=1000)
