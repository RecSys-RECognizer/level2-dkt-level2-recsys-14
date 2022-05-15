import argparse


def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument("--data_dir", default="/opt/ml/input/data/", type=str, help="data directory",)

    parser.add_argument("--asset_dir", default="asset/", type=str, help="assest directory",)

    parser.add_argument("--split_ratio", default=0.7, type=float, help="train ratio")

    parser.add_argument("--verbos_eval", default=100, type=int, help="model verbos_eval")

    parser.add_argument("--num_boost_round", default=2500, type=int, help="model num_boost_round")

    parser.add_argument("--early_stopping_rounds", default=100, type=int, help="model early_stopping_rounds")

    parser.add_argument("--threshold", default=0.5, type=float, help="predict threshold")

    parser.add_argument("--pic_dir", default="save_pic/", type=str, help="picture directory")

    parser.add_argument("--output_dir", default="output/", type=str, help="output directory")

    parser.add_argument("--model_dir", default="model/", type=str, help="model directory")

    args = parser.parse_args()

    return args

def parse_args(mode="inference"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument("--data_dir", default="/opt/ml/input/data/", type=str, help="data directory",)

    parser.add_argument("--asset_dir", default="asset/", type=str, help="assest directory",)

    parser.add_argument("--output_dir", default="output/", type=str, help="output directory")

    parser.add_argument("--model_dir", default="model/", type=str, help="model directory")

    args = parser.parse_args()

    return args
