import torch

hyperparameter_defaults  = {
    "n_epochs": 5,
    "optimizer": 'adam',
    "drop_out": 0.2
    }

sweep_config = {
  "name" : "my-sweep",
  "method" : "grid",
  "metric" : {
      "name": "valid_auc",
      "goal": "maximize"
  },
  "parameters" : {
    "n_epochs" :{
      "values": [10, 20, 30]
    },
    "optimizer":{
     "values": ['adam', 'adamW']
    },
    "drop_out":{
     "values": [0.2, 0.5, 0.7]
    },
    "device" : {
      "value": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    },
    "seed" : {
      "value":42
    },
    "data_dir" : {
      "value":'/opt/ml/input/data/'
    },
    "asset_dir" : {
      "value":'asset/'
    },
    "file_name" : {
      "value":'train_data.csv'
    },
    "model_dir" : {
      "value":'models/'
    },
    "model_name" : {
      "value":'model.pt'
    },
    "output_dir" : {
      "value":'output/'
    },
    "test_file_name" : {
      "value":'test_data.csv'
    },
    "max_seq_len" : {
      "value": 20
    },
    "num_workers" : {
      "value": 1
    },
    "hidden_dim" : {
      "value": 64
    },
    "n_layers" : {
      "value": 2
    },
    "n_heads" : {
      "value": 2
    },
    "batch_size" : {
      "value": 4
    },
    "lr" : {
      "value": 0.0001
    },
    "clip_grad" : {
      "value": 10
    },
    "patience" : {
      "value": 5
    },
    "log_steps" : {
      "value": 50
    },
    "model" : {
      "value": 'lstm'
    },
    "scheduler" : {
      "value":'plateau'
    }
  }
}