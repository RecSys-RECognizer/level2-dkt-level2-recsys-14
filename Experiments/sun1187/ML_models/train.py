import pandas as pd
import os
import random
import lightgbm as lgb
from args import parse_args
from preprocess import load_data, feature_engineering, custom_train_test_split
from trainer import train_model
from matplotlib import pyplot as plt
from inference import infer
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    args = parse_args(mode="train")

    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
            'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

    # Preprocessing
    df = load_data(args)
    df = feature_engineering(df)
    train, test = custom_train_test_split(args, df)
    print("preprocessing Done!!")

    # Train a selected model
    trained_model = train_model(args, train, test, FEATS)
    print("training Done!!")

    # Save a feature importance
    if args.model == 'lgbm_package':
        x = lgb.plot_importance(trained_model)
        if not os.path.exists(args.pic_dir):
            os.makedirs(args.pic_dir)
        plt.savefig(os.path.join(args.pic_dir, str(args.verbos_eval)+'.png'))

    # Inference
    infer(args, trained_model, FEATS)
    print("Inference Done!!")

