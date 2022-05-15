import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from wandb.lightgbm import wandb_callback
import wandb
from models import get_models, get_single_model

def train_model(args, train, test, FEATS):
    wandb.init(project="dkt_lgbm", entity="esk1")
    wandb.config.update(args)

    # split X, y
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    # get a model and train
    if args.model == 'stacking_ml':        
        S_train, model = get_models(args, train, y_train, test, y_test, FEATS)
        model.fit(S_train, y_train)
    elif args.model == 'lgbm_package':
        model = get_models(args, train, y_train, test, y_test, FEATS)
    else:
        model = get_models(args, train, y_train, test, y_test, FEATS)
        model.fit(train[FEATS], y_train)

    # Predict valid data
    preds = model.predict(test[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= args.threshold, 1, 0))
    auc = roc_auc_score(y_test, preds)

#    train_pred = model.predict(train[FEATS])
#    train_acc = accuracy_score(y_train, np.where(train_pred >= args.threshold, 1, 0))
#    train_auc = roc_auc_score(y_train, preds)

    print(f'VALID AUC : {auc} ACC : {acc}\n')
    wandb.log({
#        "train auc": train_auc,
#        "train acc": train_acc,
        "valid auc": auc,
        "valid acc": acc})

    return model

