import wandb
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os
from dkt import trainer
from dkt.dataloader import Preprocess
from dkt.utils import setSeeds

def pre_and_train(config=None):
    wandb.init(project="dkt", config=config)#vars(args))

    w_config = wandb.config

    os.makedirs(w_config.model_dir, exist_ok=True)
    preprocess = Preprocess(w_config)
    preprocess.load_train_data(w_config.file_name)
    train_data = preprocess.get_train_data()

    train_data, valid_data = preprocess.split_data(train_data)

    #sweep_id = wandb.sweep(config.sweep_config, project="dkt_lgbm", entity="esk1")
    #wandb.agent(sweep_id, trainer.run(args, train_data, valid_data), count=25)
    trainer.run(w_config, train_data, valid_data)