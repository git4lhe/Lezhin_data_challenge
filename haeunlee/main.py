import argparse
from core.utils import load_data, ExperimentSettings
from core.transforms import PipelineCreator
from core.utils import make_dir
from core.models import ModelTrainer,ModelTester

def main(args):
    xp = ExperimentSettings(args)
    if args.train_flag:
        xp.create_setting(split=False)
        pipeline = PipelineCreator(xp.numeric_features, xp.categorical_features)
        preprocessor = pipeline.make_pipeline()
        train = ModelTrainer(
            xps=xp,
            preprocessor=preprocessor,
            model="SVM",
            pipeline_save=True,
            model_save=True,
            cv=5,  # works split is FALSE
        )
        train.run_all()
    cd
    else:
        model_path = "Project_Template/2020-07-14 17:15:24.779617/2_result/model_checkpoint.joblib"
        test = ModelTester(
            data_path=args.test, target=args.target,
            model_path=model_path
        )
        test.run()


if __name__ == "__main__":

    # modify
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_flag", "-t", type=bool, default=False)
    parser.add_argument("--train", type=str, default="data/marketing_train.csv")
    parser.add_argument("--test", type=str, default="data/marketing_test.csv")
    parser.add_argument("--target", type=str, default="insurance_subscribe")
    parser.add_argument("--predict", type=str, default="pred.csv")
    parser.add_argument("--report", type=str, default="report.csv")
    args = parser.parse_args()

    main(args)
