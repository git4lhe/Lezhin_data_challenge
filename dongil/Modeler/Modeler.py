from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


class Modeler:

    def __init__(self, y_col, x_cols, mdl_name_list):
        self.y_col = y_col
        self.x_cols = x_cols
        self.mdl_name_list = mdl_name_list

        self.mdl_inst_list = list()

        for mdl_name in mdl_name_list:
            if mdl_name == 'logistic':
                clf = LogisticRegression(class_weight='balanced', solver='lbfgs')
            elif mdl_name == 'random_forest':
                clf = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight='balanced')
            else:
                clf = xgb.XGBClassifier(n_jobs=4, object='binary:logistic')

            self.mdl_inst_list.append(clf)

    def cross_validation(self, df):
        X = df[self.x_cols].to_numpy()
        y = df[self.y_col].to_numpy()
        cv = StratifiedKFold(n_splits=5, random_state=2071621)

        result_dict = dict()
        for i, clf in enumerate(self.mdl_inst_list):
            scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            result_dict[self.mdl_name_list[i]] = scores.mean()

        return result_dict
