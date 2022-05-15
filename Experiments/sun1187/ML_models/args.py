import argparse


def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )

    parser.add_argument("--split_ratio", default=0.7, type=float, help="train ratio")

    parser.add_argument("--verbos_eval", default=100, type=int, help="model verbos_eval")

    parser.add_argument("--num_boost_round", default=500, type=int, help="model num_boost_round")

    parser.add_argument("--early_stopping_rounds", default=100, type=int, help="model early_stopping_rounds")

    parser.add_argument("--threshold", default=0.5, type=float, help="predict threshold")

    parser.add_argument("--model", default="logisticregression", type=str, help="ml model")

    parser.add_argument("--voting_methd", default="hard", type=str, help="voting method")

    parser.add_argument("--ensem1", default="logisticregression", type=str, help="ensemble model1")

    parser.add_argument("--ensem2", default="gaussiannb", type=str, help="ensemble model2")

    parser.add_argument("--ensem3", default="lda", type=str, help="ensemble model3")

    parser.add_argument("--bag_model", default="gaussiannb", type=str, help="bagging model")

    parser.add_argument("--bootstrap", default=True, action="store_true", help="bagging bootstrap")

    parser.add_argument("--bootstrap_features", default=False, action="store_true", help="bagging bootstrap_features")

    parser.add_argument("--n_estimators", default=100, type=int, help="n_estimators of bagging and randomforest")

    parser.add_argument("-n_folds", default=5, type=int, help="stacking n_folds")

    parser.add_argument("--stratified", default=True, action="store_true", help="stacking stratified")

    parser.add_argument("--shuffle", default=True, action="store_true", help="stacking shuffle")

    parser.add_argument("--level2_model", default="gaussiannb", type=str, help="stacking level2 model")

    parser.add_argument("--max_samples", default=0.5, type=float, help="bagging max_samples")

    parser.add_argument("--max_features", default=1.0, type=float, help="bagging max_features")

    parser.add_argument(
        "--pic_dir", default="save_pic/", type=str, help="picture directory"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )

    args = parser.parse_args()

    return args
