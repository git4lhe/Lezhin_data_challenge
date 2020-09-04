import argparse
from haeunlee.core.source import ExperimentSettings
from haeunlee.core.transforms import PipelineCreator
from haeunlee.core.trainer import ModelTrainer
from sklearn import linear_model
from sklearn.feature_extraction.text import HashingVectorizer

def main(args):
    if args.train_flag:
        xp = ExperimentSettings(args.test, args.target)
        xp.read_data(ignore = ['10'])

        # TODO: Define Namedtuple (name, transformer, columns)
        pipeline = PipelineCreator(xp.numeric_cols, xp.str_cols, xp.ignore_cols).get_pipeline()

        # TODO: ModelTrainer
        trainer = ModelTrainer(
            preprocessor=pipeline,
            pipeline_save=True,
            model_save=True,
            cv=2,  # works split is FALSE
        )
        trainer.run_rf(xp.X, xp.y)

if __name__ == "__main__":

    # modify
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", "-t", type=bool, default=True)
    parser.add_argument(
        "--train", type=str, default="./data/lezhin_dataset_v2_training.tsv"
    )
    parser.add_argument("--test", type=str, default="./data/lezhin_dataset_v2_test.tsv")
    parser.add_argument("--target", type=str, default="1")
    parser.add_argument("--predict", type=str, default="pred.csv")
    parser.add_argument("--report", type=str, default="report.csv")
    args = parser.parse_args()

    main(args)
