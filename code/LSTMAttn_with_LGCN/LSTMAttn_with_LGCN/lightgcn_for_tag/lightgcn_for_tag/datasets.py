import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pickle 
def prepare_dataset_tag(device, basepath, verbose=True, logger=None):
    data = load_data(basepath)
    train_data_all, test_data = separate_data(data)
    
    train_data, valid_data, _, _ = train_test_split(train_data_all, train_data_all['KnowledgeTag'],
                 test_size=0.3, shuffle=True, stratify=train_data_all['KnowledgeTag'], random_state=35)
    # print(train_data_all.userID.nunique())
    # print(train_data.userID.nunique())
    # print(train_data_all.KnowledgeTag.nunique())
    # print(train_data.KnowledgeTag.nunique())
    # print( valid_data.KnowledgeTag.nunique())
    # print( valid_data.userID.nunique())
    if train_data_all.userID.nunique() == train_data.userID.nunique() and train_data_all.KnowledgeTag.nunique() == train_data.KnowledgeTag.nunique() and train_data_all.userID.nunique() == valid_data.userID.nunique() and train_data_all.KnowledgeTag.nunique() == valid_data.KnowledgeTag.nunique():
        print("Node Set okay!!!")
    else:
        print("Wrong Node Set!!!!!!!!!")

    id2index = indexing_data(data)
    with open('tag2index.pickle','wb') as fw:
        pickle.dump(id2index, fw)
    train_data_proc = process_data(train_data, id2index, device)
    valid_data_proc = process_data(valid_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)


    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)
    
    #print('number of train data', len(train_data_proc["label"]))
    #print('number of valid data', len(valid_data_proc["label"]))
    #print('number of test data', len(test_data_proc["label"]))

    return train_data_proc, valid_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)

    data = pd.concat([data1, data2])
    data= data.astype({'KnowledgeTag':'str'})
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.KnowledgeTag))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.KnowledgeTag, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.KnowledgeTag))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
