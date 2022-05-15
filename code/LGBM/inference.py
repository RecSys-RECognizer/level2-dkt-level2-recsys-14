import pandas as pd
import os
import random
from lgbm.preprocess import load_data, feature_engineering, custom_train_test_split, categorical_label_encoding, convert_time, add_diff_feature
from lgbm.preprocess import scaling
import lightgbm as lgb
from args import parse_args
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == "__main__":
    args = parse_args(mode="inference")

    # LOAD TESTDATA
    test_csv_file_path = os.path.join(args.data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv_file_path)

    # FEATURE ENGINEERING
    test_df = feature_engineering(test_df)
    test_df = categorical_label_encoding(args, test_df, is_train=False)
    test_df["Timestamp"] = test_df["Timestamp"].apply(convert_time)
    test_df = add_diff_feature(test_df)
    test_df = scaling(args, test_df, is_train=False)

    # LEAVE LAST INTERACTION ONLY
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]
    test_df.fillna(0, axis=1, inplace=True)

    # DROP ANSWERCODE
    test_df = test_df.drop(['answerCode'], axis=1)

    # SELECT Features
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_acc', 'test_mean',
     'test_sum','tag_mean', 'assessmentItemID', 'Timestamp', 'diff', 'mean']

    # MAKE PREDICTION
    model = lgb.Booster(model_file=os.path.join(args.model_dir, "lgbm_model.txt"))
    total_preds = model.predict(test_df[FEATS])

    # SAVE OUTPUT
    write_path = os.path.join(args.output_dir, "lgbm_submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
    