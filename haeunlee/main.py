import argparse
from haeunlee.core.utils import ExperimentSettings
from haeunlee.core.transforms import NumPipelineCreator, CatPipelineCreator
from sklearn.compose import ColumnTransformer
from haeunlee.core.trainer import ModelTrainer
from haeunlee.models.support_vector_machine import SVM
#
def main(args):
    xp = ExperimentSettings()
    if args.train_flag:

        xp.read_data(args.train, args.target)
        print(xp.numerical_cols, xp.categorical_cols, xp.ignore_cols)

        # TODO: Define Namedtuple (name, transformer, columns)
        num_transform = NumPipelineCreator()
        # TODO: Add clip transform

        cat_transform = CatPipelineCreator()
        preprocessor = ColumnTransformer(
            [('num_transform', num_transform.get_pipeline(), xp.numerical_cols),
             ('cat_transform', cat_transform.get_pipeline(),  xp.categorical_cols)],
            remainder='drop')

        # TODO: ModelTrainer
        train = ModelTrainer(
            xp=xp,
            preprocessor=preprocessor,
            model=SVM(),
            pipeline_save=True,
            model_save=True,
            cv=5  # works split is FALSE
        )
        train.run_all()

    # TODO: ModelTester
    # else:
    #     model_path = "Project_Template/2020-07-14 17:15:24.779617/2_result/model_checkpoint.joblib"
    #     test = ModelTester(
    #         data_path=args.test, target=args.target,
    #         model_path=model_path
    #     )
    #     test.run()


if __name__ == "__main__":

    # modify
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", "-t", type=bool, default=True)
    parser.add_argument("--train", type=str, default="./data/lezhin_dataset_v2_training.tsv")
    parser.add_argument("--test", type=str, default="./data/lezhin_dataset_v2_test.tsv")
    parser.add_argument("--target", default=1)
    parser.add_argument("--predict", type=str, default="pred.csv")
    parser.add_argument("--report", type=str, default="report.csv")
    args = parser.parse_args()

    main(args)
