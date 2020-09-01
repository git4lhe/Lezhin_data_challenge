import unittest


class TestSummary(unittest.TestCase):
    def test_numerical_(self):
        data_path = "test/titanic/titanic_train.csv"

        parser = argparse.ArgumentParser()
        parser.add_argument("--train_flag", "-t", type=bool, default=True)
        parser.add_argument("--train", type=str, default=data_path)
        parser.add_argument("--test", type=str, default=data_path)
        parser.add_argument("--target", type=str, default="Survived")
        parser.add_argument("--predict", type=str, default="pred.csv")
        parser.add_argument("--report", type=str, default="report.csv")
        args = parser.parse_args()

        main(args)


if __name__ == "__main__":
    unittest.main()
