from core.evaluation import Evaluation
import pandas as pd
import numpy as np

def write_report(y_true, y_pred, save_path):
    stat = Evaluation(y_true, y_pred)

    assert len(y_true) == len(y_pred)

    index = [i for i in range(len(y_pred))]
    pred_data = {'pred': y_pred}
    pred = pd.DataFrame(data=pred_data)

    stat_data = {"Precision": stat.precision,
                 "Recall": stat.recall,
                 "Accruacy":stat.acc,
                 "F1_score": stat.F1_score
                 }
    stat = pd.DataFrame(data=[stat_data])

    pred.to_csv(save_path + '/pred.csv',index=True)
    stat.to_csv(save_path + '/report.csv',index=False)

    print("the report saved to..", save_path)