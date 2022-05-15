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
import matplotlib.pyplot as pl
from sklearn.preprocessing import MinMaxScaler


def tag_weight(train, test):
    tag_cnt = train['KnowledgeTag'].value_counts()
    mm = MinMaxScaler()
    tag_weight = mm.fit_transform(tag_cnt.values.reshape(-1, 1)).reshape(1,-1)[0]

    dict_tag_weight = {j : tag_weight[i] for i, j in enumerate(tag_cnt.index)}
    train['tag_weight'] = train['KnowledgeTag'].apply(lambda x : dict_tag_weight[x])
    test['tag_weight'] = test['KnowledgeTag'].apply(lambda x : dict_tag_weight[x])
    
    return train, test

def recent_acc(df):
    shift_size = 5
    df = df.sort_values(by=['userID'])
    df_temp = df.copy()

    # 새롭게 이력이 시작되는 유저 구함
    user_start_pos = df['userID'].diff() > 0
    df_temp['previous_answer_count'] = df_temp.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    df_temp['shift_previous_answer_count'] = df_temp.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    df_temp.loc[user_start_pos, ['previous_answer_count', 'shift_previous_answer_count']] = 0
    df_temp['shift_previous_answer_count'] = df_temp['shift_previous_answer_count'].shift(shift_size)
    df_temp['temp'] = len(df_temp) * [1]
    df_temp['previous_problem_count'] = df_temp.groupby('userID')['temp'].cumsum().shift(fill_value=0)
    df_temp['previous_problem_count'] = df_temp['previous_problem_count'].apply(lambda x: shift_size if x > shift_size else x)
    df_temp.loc[user_start_pos, ['previous_problem_count']] = 0

    df_temp['shift_previous_answer_count'] = df_temp.apply(lambda x: 0 if x['previous_problem_count'] < shift_size else x['shift_previous_answer_count'], axis=1)
    df_temp['count'] = df_temp['previous_answer_count'] - df_temp['shift_previous_answer_count']
    df['user_recent_acc'] = (df_temp['count'] / df_temp['previous_problem_count']).fillna(0)
    
    return df

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, ratio=0.8, split=True):
    random.seed(42)
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)

    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test

def feature_engineering(df):
    # scaler = MinMaxScaler()
    def convert_time(s):
        timestamp = time.mktime(
            datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        )
        return int(timestamp)

    df["Timestamp"] = df["Timestamp"].apply(convert_time)

    # --- diff
    # df['diff'] = df.sort_values(['userID','Timestamp']).groupby('userID')['Timestamp'].diff(periods=1).apply(abs)
    df['diff'] = df.sort_values(['userID','Timestamp']).groupby('userID')['Timestamp'].diff()

    # nan은 -1
    # 600(10분) 이상이면 다 600
    df['diff'].fillna(-1, inplace=True)
    idx = df[df['diff'] >= 600].index
    df.loc[idx, 'diff'] = 600

    # # --- mean 태그별 문제 풀이 평균 시간
    tmp= df[df['diff']>=0]
    correct_k= tmp.groupby(['KnowledgeTag'])['diff'].agg(['mean'])
    df= pd.merge(df, correct_k, on=['KnowledgeTag'], how= 'left')
    # df['mean']= scaler.fit_transform(df['mean'].values.reshape(-1, 1)).reshape(-1) # minmax scaling


    # --- before_tag 이전 태그 문제 풀이 여부
    df2= df.sort_values(['userID','KnowledgeTag','Timestamp'])
    df2.reset_index(inplace=True, drop=True)
    df2['cumsum']= df2.groupby(['userID','KnowledgeTag'])['answerCode'].cumsum()
    df2['temp']= 1
    df2['seq']= df2.groupby(['userID','KnowledgeTag'])['temp'].cumsum()
    df2.drop(['temp'], axis=1, inplace=True)
    df2['cumsum'] -= df2['answerCode']

    df2['seq'] -= 1
    df2['before_tag']= df2['cumsum']/df2['seq']

    tag_avg= dict(df2.groupby(['KnowledgeTag'])['answerCode'].mean())
    def match_avg(x):
        if x>1:
            return tag_avg[x]
        else:
            return x

    df2['before_tag'].fillna(df2.KnowledgeTag, inplace=True)
    df2['before_tag']= df2['before_tag'].apply(match_avg)
    df2.loc[df2[df2['before_tag'] >= 0.5].index,'before_tag'] = 1
    df2.loc[df2[df2['before_tag'] < 0.5].index,'before_tag'] = 0
    
    df= pd.merge(df2[['userID','assessmentItemID','Timestamp','before_tag']],df, on=['userID','assessmentItemID','Timestamp'])

    # # --- user_correct_answer
    df2= df.sort_values(by=['userID','Timestamp'])
    df2['user_correct_answer'] = df2.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df2['user_correct_answer'].fillna(0)
    
    # # --- user_acc
    df2['user_total_answer'] = df2.groupby('userID')['answerCode'].cumcount()
    df2['user_acc'] = df2['user_correct_answer']/df2['user_total_answer']
    
    # df2['user_correct_answer']= scaler.fit_transform(df2['user_correct_answer'].values.reshape(-1, 1)).reshape(-1)

    # # test mean, sum
    correct_t = df2.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    df2 = pd.merge(df2, correct_t, on=['testId'], how="left")
    # df2['test_mean']= scaler.fit_transform(df2['test_mean'].values.reshape(-1,1)).reshape(-1) 
    # df2['test_sum']= scaler.fit_transform(df2['test_sum'].values.reshape(-1,1)).reshape(-1) 

    df= pd.merge(df2[['userID','assessmentItemID','Timestamp','user_correct_answer', 'user_acc', 'test_mean', 'test_sum']],df, on=['userID','assessmentItemID','Timestamp'])
    
    # # --- test_seq
    df2= df.sort_values(['userID','testId','Timestamp'])
    df2.reset_index(inplace=True, drop=True)
    df2['test_seq']= df2.groupby(['userID','testId']).cumcount()
    # self.args.num_test_seq= df2['test_seq'].max()+2 # 개수 + 패딩
    df= pd.merge(df2[['userID','assessmentItemID','Timestamp','test_seq']], df, on=['userID','assessmentItemID','Timestamp'])

    # # --- Timestamp 스케일링
    # df['Timestamp']= scaler.fit_transform(df['Timestamp'].values.reshape(-1,1)).reshape(-1) 

    df['item']= df['assessmentItemID']       

    def percentile(s):
        return np.sum(s) / len(s)
    prob_groupby = df.groupby('assessmentItemID').agg({
    'userID': 'count',
    'answerCode': percentile})
    num_mean= prob_groupby.userID.mean()

    # # 태그 유형별 정답률
    # din_tag= np.load('/opt/ml/input/data/din_tag.npy')
    # in_tag= np.load('/opt/ml/input/data/in_tag.npy')

    # tag= df[['assessmentItemID','KnowledgeTag']]
    # tag= tag.drop_duplicates('assessmentItemID', keep='first')
    # prob_groupby= pd.merge(prob_groupby, tag, on='assessmentItemID', how='right')

    # up_avg= prob_groupby[prob_groupby.userID >= num_mean]
    # down_avg= prob_groupby[prob_groupby.userID < num_mean]

    # up_avg1= pd.DataFrame(up_avg.groupby('KnowledgeTag').agg({'answerCode':'mean'}))
    # up_avg1.reset_index(inplace=True)
    # up_avg1.rename({'answerCode':'tagAr'}, axis=1, inplace=True)
    # down_avg1= pd.DataFrame(down_avg.groupby('KnowledgeTag').agg({'answerCode':'mean'}))
    # down_avg1.reset_index(inplace=True)
    # down_avg1.rename({'answerCode':'tagAr'}, axis=1, inplace=True)

    # # ## 로그 스케일링
    # up_avg1.loc[up_avg1[up_avg1.KnowledgeTag.isin(in_tag)].index,'tagAr']=up_avg1[up_avg1.KnowledgeTag.isin(in_tag)]['tagAr'].apply(np.log1p)
    # down_avg1.loc[down_avg1[down_avg1.KnowledgeTag.isin(din_tag)].index,'tagAr']=down_avg1[down_avg1.KnowledgeTag.isin(din_tag)]['tagAr'].apply(np.log1p)

    # new_up_avg= pd.merge(up_avg, up_avg1,on='KnowledgeTag')
    # new_down_avg= pd.merge(down_avg, down_avg1,on='KnowledgeTag')

    # new_down_avg.tagAr *=new_down_avg.userID/1000
    # new_down_avg.tagAr *=new_down_avg.userID/1000

    # new_= pd.concat([new_up_avg, new_down_avg], axis=0)
    # df= pd.merge(new_[['assessmentItemID','KnowledgeTag']], df, on=['assessmentItemID','KnowledgeTag'], how='left')

    item_group = df.groupby('assessmentItemID')['answerCode'].mean()
    dict_item_mean = dict(item_group)
    df['item_acc'] = df['assessmentItemID'].apply(lambda x : dict_item_mean[x])
    
    # def item_acc_cate(x):
    #     if x >= 0.75: return 2
    #     elif x >= 0.5: return 1
    #     else: return 0
    # df['item_acc'] = df['item_acc'].apply(item_acc_cate)
    # df['item_acc']= scaler.fit_transform(df['item_acc'].values.reshape(-1, 1)).reshape(-1) # minmax scaling

    ## 최근 5개 문제 정답률
    # 새롭게 이력이 시작되는 유저 구함
    # shift_size = 5
    # df = df.sort_values(by=['userID'])
    # df_temp = df.copy()

    # # 새롭게 이력이 시작되는 유저 구함
    # user_start_pos = train_df['userID'].diff() > 0
    # df_temp['previous_answer_count'] = df_temp.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    # df_temp['shift_previous_answer_count'] = df_temp.groupby('userID')['answerCode'].cumsum().shift(fill_value=0)
    # df_temp.loc[user_start_pos, ['previous_answer_count', 'shift_previous_answer_count']] = 0
    # df_temp['shift_previous_answer_count'] = df_temp['shift_previous_answer_count'].shift(shift_size)
    # df_temp['temp'] = len(df_temp) * [1]
    # df_temp['previous_problem_count'] = df_temp.groupby('userID')['temp'].cumsum().shift(fill_value=0)
    # df_temp['previous_problem_count'] = df_temp['previous_problem_count'].apply(lambda x: shift_size if x > shift_size else x)
    # df_temp.loc[user_start_pos, ['previous_problem_count']] = 0

    # df_temp['shift_previous_answer_count'] = df_temp.apply(lambda x: 0 if x['previous_problem_count'] < shift_size else x['shift_previous_answer_count'], axis=1)
    # df_temp['count'] = df_temp['previous_answer_count'] - df_temp['shift_previous_answer_count']
    # df['user_recent_acc'] = (df_temp['count'] / df_temp['previous_problem_count']).fillna(0)

    knowledgetags = df.KnowledgeTag
    knowledgetag_stroke = np.zeros(knowledgetags.shape)

    for i, k in enumerate(knowledgetags):
        if i == 0:
            continue

        if k == knowledgetags[i-1]:
            knowledgetag_stroke[i] = knowledgetag_stroke[i-1] + 1

    df['knowledgetag_stroke'] = knowledgetag_stroke

    return df


def scaling(df, ms=None, tms=None, ts=None, tss=None, is_train=True):

    if is_train:
        mean_scaler = MinMaxScaler()
        test_mean_scalar = MinMaxScaler()
        tmestamp_scalar = MinMaxScaler()
        test_sum_scalar = MinMaxScaler()

        df['mean'] = mean_scaler.fit_transform(df['mean'].values.reshape(-1, 1))
        df['test_mean'] = test_mean_scalar.fit_transform(df['test_mean'].values.reshape(-1, 1))
        df['Timestamp'] = tmestamp_scalar.fit_transform(df['Timestamp'].values.reshape(-1, 1))
        df['test_sum'] = test_sum_scalar.fit_transform(df['test_sum'].values.reshape(-1, 1))
        
        return mean_scaler, test_mean_scalar, tmestamp_scalar, test_sum_scalar, df

    else:
        df['mean'] = ms.transform(df['mean'].values.reshape(-1, 1))
        df['test_mean'] = tms.transform(df['test_mean'].values.reshape(-1, 1))
        df['Timestamp'] = ts.transform(df['Timestamp'].values.reshape(-1, 1))
        df['test_sum'] = tss.transform(df['test_sum'].values.reshape(-1, 1))
        
        return df