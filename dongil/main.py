from DataCollector.DataCollector import DataCollector
from DataCleanser.DataCleanser import DataCleanser
from FeatureController.PurchaseRatioMaker import PurchaseRatioMaker

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


if __name__ == '__main__':
    dataCollector = DataCollector(data_folder_path='../data')
    df = dataCollector.read_csv_to_df('lezhin_dataset_v2_training.tsv')
    df.columns = [str(x) for x in df.columns]

    y_col = '1'
    x_col_list = [x for x in df.columns if x != y_col]

    impute_col_list = ['156', '159', '164', '165']
    fill_value = 2.0

    purchase_ratio_col_list = ['7', '8']
    count_threshold = 100
    new_col_prefix = 'purchase_ratio'

    column_trans = ColumnTransformer(
        [('simple_impute', SimpleImputer(strategy='constant', fill_value=fill_value), impute_col_list),
         ('purchase_ratio', PurchaseRatioMaker(y_col=y_col,
                                               count_threshold=count_threshold,
                                               new_col_prefix=new_col_prefix), purchase_ratio_col_list)],
        remainder='passthrough')

    na_threshold = 0.1
    cate_threshold = 1
    dataCleanser = DataCleanser(na_threshold=na_threshold, cate_threshold=cate_threshold)

    pipe = Pipeline(
        steps=[('column_trans', column_trans),
               ('data_cleanser', dataCleanser),
               ('scaler', StandardScaler()),
               ('random_foest', RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced'))],
        verbose=True
    )

    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(pipe, df[x_col_list], df[y_col], cv=cv, scoring='roc_auc')
    print(scores.mean())

    # result = pipe.fit_transform(X=df[x_col_list], y=df[y_col])

    # dataVisualizer = DataVisualizer(y_col=y_col, save_path='../img')
    # for x_col in range(152, 168):
    #     dataVisualizer.na_col_plot(_df=df, x_col=x_col)
    # for x_col in (7, 8, 10):
    #     dataVisualizer.str_col_plot(_df=df, x_col=x_col)

    # impute_col_list = [156, 159, 164, 165]
    # featureController = FeatureController(y_col=y_col)
    # featureController.custom_imputation(df=df, impute_col_list=impute_col_list)
    # df = featureController.make_purchase_ratio_column(df=df, x_col=7)
    #
    # dataCleanser = DataCleanser(y_col=y_col)
    # df = dataCleanser.remove_na_col(df, threshold=0.1)
    # df = dataCleanser.remove_str_col(df)
    # df = dataCleanser.remove_cate_col(df, threshold=1)
    #
    # x_cols = [x for x in df.columns if x != y_col]
    # scaler = Scaler(y_col=y_col, x_cols=x_cols, scaler_method='MinMax')
    # scaler.fit_scaler(df)
    # df = scaler.transform_df(df)

    # mdl_name_list = ['logistic', 'random_forest', 'xgboost']
    # modeler = Modeler(y_col=y_col, x_cols=x_cols, mdl_name_list=mdl_name_list)
    # result = modeler.cross_validation(df)

    # print(result)

