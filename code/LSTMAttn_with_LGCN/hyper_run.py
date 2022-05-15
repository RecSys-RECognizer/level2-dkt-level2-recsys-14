from hyperopt import *
import hyperopt
import matplotlib.pyplot as pl
import easydict

import os
import pandas as pd
import torch
import random
from args import parse_args
from LSTMAttn_with_LGCN import trainer
from LSTMAttn_with_LGCN.dataloader import Preprocess
from LSTMAttn_with_LGCN.utils import setSeeds

# 최적화 할 함수
def objective_function(space):
    """
    space 예시 {'batch_size': 64, 'lr': 0.00010810929882981193, 'n_layers': 1}
    """

    # args가 dict으로 건네지기 때문에 easydict으로 바꿔준다
    args = space['args']

    # 하이퍼파라메타 값 변경
    args.n_layers = space['n_layers']
    args.n_heads = space['n_heads']
    args.drop_out = space['drop_out']
    args.max_seq_len = space['seq_len']
    args.batch_size= space['batch_size']

    # seed 설정
    setSeeds(42)
    
    report = trainer2.run(args, train, valid)

    best_auc = report

    return -best_auc

def trials_to_df(trials, space, best):
    # 전체 결과
    rows = []
    keys = list(trials.trials[0]['misc']['vals'].keys())

    # 전체 실험결과 저장
    for trial in trials:
        row = {}

        # tid
        tid = trial['tid']
        row['experiment'] = str(tid)
        
        # hyperparameter 값 저장
        vals = trial['misc']['vals']
        hparam = {key: value[0] for key, value in vals.items()}

        # space가 1개 - 값을 바로 반환
        # space가 다수 - dict에 값을 반환
        hparam = hyperopt.space_eval(space, hparam)

        if len(keys) == 1:
            row[keys[0]] = hparam
        else:
            for key in keys:
                row[key] = hparam[key]

        # metric
        row['metric'] = abs(trial['result']['loss'])
        
        # 소요 시간
        row['time'] = (trial['refresh_time'] - trial['book_time']).total_seconds() 
        
        rows.append(row)

    experiment_df = pd.DataFrame(rows)
    
    # best 실험
    row = {}
    best_hparam = hyperopt.space_eval(space, best)

    if len(keys) == 1:
        row[keys[0]] = best_hparam
    else:
        for key in keys:
            row[key] = best_hparam[key]
    row['experiment'] = 'best'

    best_df = pd.DataFrame([row])

    # best 결과의 auc / time searching 하여 찾기
    search_df = pd.merge(best_df, experiment_df, on=keys)
    
    # column명 변경
    search_df = search_df.drop(columns=['experiment_y'])
    search_df = search_df.rename(columns={'experiment_x': 'experiment'})

    # 가장 좋은 metric 결과 중 가장 짧은 시간을 가진 결과를 가져옴 
    best_time = search_df.time.min()
    search_df = search_df.query("time == @best_time")

    df = pd.concat([experiment_df, search_df], axis=0)

    return df

def main(args):

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)
    global train 
    train= train_data
    global valid 
    valid= valid_data

    # 탐색 공간
    space = {
        'n_layers': hp.choice('n_layers', [2,3,4]),
        'n_heads': hp.choice('n_heads', [2,8,16]),
        'drop_out': hp.choice('drop_out', [0.5,0.2]),
        'seq_len': hp.choice('seq_len', [20,100,256,1024]),
        'batch_size':hp.choice('batch_size',[32,64,128]),
        'args': args
    }



    # 최적화
    trials = Trials()
    best = fmin(
                fn=objective_function,  # 최적화 할 함수
                space=space,          # Hyperparameter 탐색 공간
                algo=tpe.suggest,       # Tree-structured Parzen Estimator (TPE)
                max_evals=60,           # 10번 시도
                trials=trials
                )

    df = trials_to_df(trials, space, best)
    df.to_csv('hyper_lstm.csv', index=False)

if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)

