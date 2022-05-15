from audioop import reverse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import os
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from dataset import *
from config import *
import warnings
warnings.filterwarnings('ignore')

print("Data Load Start [1/7]")
train_df = pd.read_csv(CFG.basepath + "train_data.csv")
test_df = pd.read_csv(CFG.basepath + "test_data.csv")
print("Data Load Done! [1/7]\n")

print("Feature Engineering Start [2/7]")
train_df_fe = feature_engineering(train_df)
test_df_fe = feature_engineering(test_df)
print("Feature Engineering Done! [2/7]\n")

print("Scaling Start [3/7]")
mean_sc, test_mean_sc, time_sc, test_sum_sc, train_df_fe2 = scaling(train_df_fe, is_train=True)
test_df_fe2 = scaling(test_df_fe, mean_sc, test_mean_sc, time_sc, test_sum_sc, is_train=False)
print("Scaling Done! [3/7]\n")

print("train, valit set split Start [4/7]")
inference = test_df_fe2.loc[test_df_fe2['answerCode'] == -1]
infer_test = test_df_fe2.loc[test_df_fe2['answerCode'] != -1]

train, test = custom_train_test_split(train_df_fe2)

print("train, valid set split Done! [4/7]\n")

print("Compose data pool for CatBoost [5/7]")
categorical_cols =  ['assessmentItemID', 'before_tag']
TARGET_COL = 'answerCode'
features = ['assessmentItemID', 'answerCode', 'user_acc', 'test_mean',
                'before_tag', 'diff']

# Train Pool

train = train[features]
# train = train[['answerCode', 'user_acc', 'testId', 'test_mean',
#                 'before_tag', 'diff', 'item_acc']]
train[categorical_cols] = train[categorical_cols].fillna("")
train['diff'] = train['diff'].astype(int)
train['before_tag'] = train['before_tag'].astype(int)
# train['user_recent_acc'] = train['user_recent_acc'].astype(int)

X_train = train.drop([TARGET_COL],axis=1)
y_train = train[TARGET_COL]
train_pool = Pool(data=X_train,label = y_train,cat_features=categorical_cols)

# test_pool
test = test[features]
test[categorical_cols] = test[categorical_cols].fillna("")
# test[['before_tag', 'diff']] = test[['before_tag', 'diff']].astype(int)
test['diff'] = test['diff'].astype(int)
test['before_tag'] = test['before_tag'].astype(int)
# test['user_recent_acc'] = test['user_recent_acc'].astype(int)
# test.drop(columns=['Timestamp'], inplace=True)

X_test = test.drop([TARGET_COL],axis=1)
y_test = test[TARGET_COL]
test_pool = Pool(data=X_test,label = y_test,cat_features=categorical_cols)

# infer_pool
# df_infer = feature_engineering(df_infer)
# categorical_cols =  ['before_tag', 'testId']
inference = inference[features]
inference[categorical_cols] = inference[categorical_cols].fillna("")
inference['diff'] = inference['diff'].astype(int)
# inference[['before_tag', 'diff']] = inference[['before_tag', 'diff']].astype(int)
inference['before_tag'] = inference['before_tag'].astype(int)
# inference['user_recent_acc'] = inference['user_recent_acc'].astype(int)
# inference.drop(columns=['Timestamp'], inplace=True)

X_infer = inference.drop([TARGET_COL],axis=1)
y_infer = inference[TARGET_COL]
infer_pool = Pool(data=X_infer,label = y_infer,cat_features=categorical_cols)

# infer_test_pool
infer_test1 = infer_test[features]
infer_test1[categorical_cols] = infer_test1[categorical_cols].fillna("")
infer_test1['diff'] = infer_test1['diff'].astype(int)
# inference[['before_tag', 'diff']] = inference[['before_tag', 'diff']].astype(int)
infer_test1['before_tag'] = infer_test1['before_tag'].astype(int)
# infer_test1['user_recent_acc'] = infer_test1['user_recent_acc'].astype(int)

X_infer_test = infer_test1.drop([TARGET_COL],axis=1)
y_infer_test = infer_test1[TARGET_COL]
infer_test_pool = Pool(data=X_infer_test,label = y_infer_test,cat_features=categorical_cols)
print("Compose data pool for CatBoost Done! [5/7]\n")

print("Train Start [6/7]")
print('params :', CFG.params)
print()
model_basic = CatBoostClassifier(**CFG.params)#,learning_rate=0.1, task_type="GPU",)
# model_basic = CatBoostClassifier(verbose=50)#,learning_rate=0.1, task_type="GPU",)
model_basic.fit(train_pool, eval_set=test_pool, use_best_model=True)
print("Train Done! [6/7]\n")
print(model_basic.get_best_score())
print()

print("feature importances")
importances = pd.Series(model_basic.feature_importances_, index = train_pool.get_feature_names())
importances = importances.sort_values(ascending = False)
print(importances)
print()

print("============Scores============")
print("train ACC :", model_basic.score(train_pool))
print("valid ACC :", model_basic.score(test_pool))

pred = model_basic.predict_proba(test_pool)[:,1]
print("valid AUC :", roc_auc_score(y_test, pred))

pred1 = model_basic.predict_proba(infer_test_pool)[:,1]
print("test_infer AUC :", roc_auc_score(y_infer_test, pred1))
print()

print("valid set 평균, 최대값, 최소값")
print("평균 :", pred.mean())
print("최대 :", pred.max())
print("최소 :", pred.min())

print("\ninfer_test set 평균, 최대값, 최소값")
print("평균 :", pred1.mean())
print("최대 :", pred1.max())
print("최소 :", pred1.min())
print("=============================\n")

print("inference Start [7/7]")
infer_cbc = model_basic.predict_proba(infer_pool)[:,1]
print("\ninfer set 평균, 최대값, 최소값")
print("평균 :", infer_cbc.mean())
print("최대 :", infer_cbc.max())
print("최소 :", infer_cbc.min())

print("inference Done [7/7]\n")

output = pd.DataFrame({"id":range(744), 'prediction':infer_cbc})
print("0.5 이상인 것의 개수 :", len(output[output.prediction >= 0.5]),'개')

output.to_csv("CatBoost.csv", index=False)
print("output 저장 완료.")