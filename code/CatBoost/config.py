# ====================================================
# CFG
# ====================================================
class CFG:
    # train
    params = {
          'iterations':5000,
          'random_seed':63,
          'learning_rate':0.01,
          # 'max_depth' : 10,
          'eval_metric' : 'AUC',
          'loss_function':'CrossEntropy',
          'early_stopping_rounds':100,
          'use_best_model': True,
          'task_type':"GPU",
          'verbose':100,
          }

    # data
    basepath = "/opt/ml/input/data/"

    # dump
    output_dir = "./output/"