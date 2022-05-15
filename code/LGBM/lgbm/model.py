import wandb
from wandb.lightgbm import wandb_callback
import wandb
import lightgbm as lgb
import numpy as np
import os

def get_model(args, train, y_train, test, y_test, FEATS):
    
    lgb_train = lgb.Dataset(train[FEATS], y_train)
    lgb_test = lgb.Dataset(test[FEATS], y_test)

    model = lgb.train(
            {'objective': 'binary'}, 
            lgb_train,
            valid_sets=[lgb_train, lgb_test],
            verbose_eval=args.verbos_eval,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            callbacks=[wandb_callback()]
        )

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_path = os.path.join(args.model_dir, "lgbm_model.txt")
    model.save_model(model_path)

    return model

