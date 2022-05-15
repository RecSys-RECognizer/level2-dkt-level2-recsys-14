import pandas as pd
import os
import random
import lightgbm as lgb
from args import parse_args
from lgbm.preprocess import load_data, feature_engineering, custom_train_test_split, categorical_label_encoding, convert_time, add_diff_feature
from lgbm.trainer import train_model
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    args = parse_args(mode="train")

    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_acc', 'test_mean',
     'test_sum','tag_mean', 'assessmentItemID', 'Timestamp', 'diff', 'mean']

    # Preprocessing
    print("Preprocessing!")
    df = load_data(args)
    df = feature_engineering(df)
    df = categorical_label_encoding(args, df, is_train=True)
    df["Timestamp"] = df["Timestamp"].apply(convert_time)
    df = add_diff_feature(df)

    train, test = custom_train_test_split(args, df)
    print("Done Preprocessing!!")

    # Train a selected model
    trained_model = train_model(args, train, test, FEATS)
    print("Done training!!")

    # Save a feature importance
    x = lgb.plot_importance(trained_model)
    if not os.path.exists(args.pic_dir):
        os.makedirs(args.pic_dir)
    plt.savefig(os.path.join(args.pic_dir, 'lgbm_feature_importance.png'))
    print("Done!")


