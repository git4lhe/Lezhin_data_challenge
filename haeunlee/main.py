import argparse
from haeunlee.core.source import ExperimentSettings
from haeunlee.core.transforms import PipelineCreator
from haeunlee.core.trainer import ModelTrainer

def main(args):
    if args.train_flag:
        xp = ExperimentSettings(args.train, args.target)
        xp.read_data()
        xp.set_ignore([i for i in range(152,168)])

        # TODO: Define Namedtuple (name, transformer, columns)
        pipeline = PipelineCreator(xp.numeric_cols, xp.cat_cols, xp.ignore)

        # TODO: ModelTrainer
        train = ModelTrainer(
            xp=xp,
            preprocessor=pipeline.get_pipeline(),
            pipeline_save=True,
            model_save=True,
            cv=3,  # works split is FALSE
        )
        train.run_all_rf()
        train.run_all_ridge()
        train.run_all_xgboost()

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
    parser.add_argument(
        "--train", type=str, default="./data/lezhin_dataset_v2_training.tsv"
    )
    parser.add_argument("--test", type=str, default="./data/lezhin_dataset_v2_training.tsv")
    parser.add_argument("--target", type = str, default='1')
    parser.add_argument("--predict", type=str, default="pred.csv")
    parser.add_argument("--report", type=str, default="report.csv")
    args = parser.parse_args()

    main(args)
