from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from vecstack import stacking
import wandb
from wandb.lightgbm import wandb_callback
import wandb
import lightgbm as lgb
import numpy as np


def get_models(args, train, y_train, test, y_test, FEATS):
    if args.model == 'voting':
        model1 = get_single_model(args.ensem1)
        model2 = get_single_model(args.ensem2)
        model3 = get_single_model(args.ensem3)

        estimators = [
            ('model1', model1),
            ('model2', model2),
            ('model3', model3)
        ]
        model = VotingClassifier(estimators=estimators, voting=args.voting_method)
        return model

    elif args.model == 'bagging':
        model = BaggingClassifier(
            base_estimator =args.bag_model,
            n_estimators = args.n_estimators,
            bootstrap = args.boostrap,
            max_samples = args.max_samples,
            bootstrap_features = args.bootstrap_features,
            max_features = args.max_features,
            random_state = args.seed
        )
        return model

    elif args.model == 'lgbm_package':
         #apply lgb dataset
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
        return model

    elif args.model == 'stacking_ml':
        # 출처 - https://towardsdatascience.com/automate-stacking-in-python-fc3e7834772e
        ### oof stacking
        models = [get_single_model(args.ensem1),
                get_single_model(args.ensem2),
                get_single_model(args.ensem3)]
        # 1st level training
        S_train, S_test = stacking(
            models,
            train[FEATS], y_train, test[FEATS],
            n_folds = args.n_folds,
            stratified = args.stratified,
            shuffle = args.shuffle
        )
        # 2nd level training
        meta_model = args.level2_model()
        meta_model.fit(S_train, y_train)
        return meta_model

    else:
        model = get_single_model(args.model)
        return model


def get_single_model(model_str):
    if model_str == 'logisticregression':
        return LogisticRegression()
    elif model_str == 'decisiontreeclassifier':
        return DecisionTreeClassifier()
    elif model_str == 'gaussiannb':
        return GaussianNB()
    elif model_str == 'lda':
        return LinearDiscriminantAnalysis()
    elif model_str == 'qda':
        return QuadraticDiscriminantAnalysis()
    elif model_str == 'svc':
        return SVC()
    elif model_str == 'kmeans':
        return KMeans(n_clusters=2)
    elif model_str == 'randomforest':
        return RandomForestClassifier()
    elif model_str == 'extratreesclassifier':
        return ExtraTreesClassifier()
    elif model_str == 'adaboostclassifier':
        return AdaBoostClassifier()
    elif model_str == 'gradientboostingclassifier':
        return GradientBoostingClassifier()
    elif model_str == 'catboostclassfier':
        return CatBoostClassifier()
    elif model_str == 'lgbm':
        return LGBMClassifier()
    else:
        raise ValueError("Check model name")

    