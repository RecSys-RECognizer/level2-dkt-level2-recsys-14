import os
import torch
import wandb
from args import parse_args
import warnings
warnings.filterwarnings("ignore")
import argparse
from train_all import pre_and_train

def main():
    #wandb.login()
    parser = argparse.ArgumentParser(description='To Resume Sweep')
    parser.add_argument('--resume', type=str, default=None,
                    help='Input Your Sweep ID')

    parser.add_argument('--config', type=str, default="config",
                    help='Import Your Config')
    args = parser.parse_args()

    if args.resume is not None:
        sweep_id = args.resume
    else:
        if args.config == "config":
            from dkt import config as config
        else:
            print("Config File 제대로 입력하세요")
            os._exit(0)

        sweep_id = wandb.sweep(config.sweep_config, project="dkt_lgbm", entity="esk1")
    
    wandb.agent(sweep_id, pre_and_train, count=50)



if __name__ == '__main__':
    print("LSTM SWEEP START!")
    main()
