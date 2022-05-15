import pandas as pd
import os
import random
from preprocess import feature_engineering

def infer(args, model, FEATS):
    # LOAD TESTDATA
    test_csv_file_path = os.path.join(args.data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv_file_path)

    # FEATURE ENGINEERING
    test_df = feature_engineering(test_df)

    # LEAVE LAST INTERACTION ONLY
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

    # DROP ANSWERCODE
    test_df = test_df.drop(['answerCode'], axis=1)

    # MAKE PREDICTION
    total_preds = model.predict(test_df[FEATS])

    # SAVE OUTPUT
    write_path = os.path.join(args.output_dir, args.model + "_" + "submission.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
    