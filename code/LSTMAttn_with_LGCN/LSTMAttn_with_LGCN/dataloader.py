import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

        with open('/opt/ml/input/code/dkt/dkt/lightgcn/id2index.pickle', 'rb') as fr:
            self.id2idx = pickle.load(fr)
        with open('/opt/ml/input/code/dkt/dkt/lightgcn_for_tag/tag2index.pickle', 'rb') as fr:
            self.tag2idx = pickle.load(fr)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __change(self, x):
        result= self.id2idx[x]
        return result
            
    def __change_tag(self, x):
        result= self.tag2idx[x]
        return result

    def __preprocessing(self, df, is_train=True):

        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        df['item']= df['item'].apply(self.__change)
        df['tag_for_embed']= df['tag_for_embed'].apply(self.__change_tag)

        return df

    def __feature_engineering(self, df):
        scaler= MinMaxScaler()
        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        # --- diff
        df['diff'] = df.sort_values(['userID','Timestamp']).groupby('userID')['Timestamp'].diff(periods=-1).apply(abs)

        # nan은 -1
        # 600(10분) 이상이면 다 600
        df['diff'].fillna(-1, inplace=True)
        idx = df[df['diff'] >= 600].index
        df.loc[idx, 'diff'] = 600

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

        df['item']= df['assessmentItemID']       
        df['tag_for_embed']= df['KnowledgeTag'].astype(str)


        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag", "diff", "before_tag", 'item', 'tag_for_embed', 'lgbm','catboost']
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["diff"].values,
                    r["before_tag"].values,
                    r['item'].values,
                    # r["userID"].values,
                    r["tag_for_embed"].values,
                    r["lgbm"].values,
                    r["catboost"].values
                )
            )
        )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        test, question, tag, correct, diff, before_tag, item,tag_for_embed, lgbm, catboost = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9]
        cate_cols = [test, question, tag, correct, diff, before_tag, item, tag_for_embed, lgbm, catboost]


        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1 #마지막 seq에서 끝까지

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
