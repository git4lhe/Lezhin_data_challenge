import argparse
from haeunlee.core.source import ExperimentSettings
from haeunlee.core.transforms import PipelineCreator
from haeunlee.core.trainer import ModelTrainer
from sklearn import linear_model

def main(args):
    if args.train_flag:
        xp = ExperimentSettings(args.test, args.target)
        xp.read_data()
        xp.set_ignore([str(i) for i in range(152, 168)])
        print(xp.numeric_cols, xp.cat_cols, xp.target_col, type(xp.target_col))

        # TODO: Define Namedtuple (name, transformer, columns)
        pipeline = PipelineCreator(xp.numeric_cols, xp.cat_cols, xp.ignore)
        X_t = pipeline.get_pipeline().fit_transform(xp.X)

        # TODO: ModelTrainer
        trainer = ModelTrainer(
            xp=xp,
            xt=X_t,
            preprocessor=pipeline.get_pipeline(),
            pipeline_save=True,
            model_save=True,
            cv=2,  # works split is FALSE
        )

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
