import os
import pandas as pd
import torch
import wandb
from args import parse_args
from LSTMAttn_with_LGCN import trainer
from LSTMAttn_with_LGCN.dataloader import Preprocess
from LSTMAttn_with_LGCN.utils import setSeeds
import numpy as np

def main(args):
    if args.use_wandb == 'o':
        wandb.login()

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)
    # np.save('/opt/ml/input/code/dkt/output/valid_data.npy',valid_data)  # for 후시분석

    if args.use_wandb== 'o':
        wandb.init(project="dkt", config=vars(args))
        wandb.run.name = args.model_name
    trainer2.run(args, train_data, valid_data)


if __name__ == "__main__":
    args = parse_args(mode="train")
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
