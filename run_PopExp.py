import os, multiprocessing
# os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
# os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
import sys
sys.argv = [sys.argv[0]] 
import pandas as pd
import random
import torch
import argparse
import numpy as np
from pathlib import Path
import logging
import wandb
import json

from utils.Hyperopt import Hyperoptimizer
from utils.logger import LoggerMixin, init_fancy_logging
from exp.exp_pop_pre_train import Exp_Pop_Main
from exp.exp_patient_specific import Exp_PS_Main
from exp.exp_data_specific import Exp_DS_Main
from exp.exp_xgboost import Exp_xgboost
from exp.exp_test import Exp_Test
from main_configs import Benchmark_args

log_file = "/mnt/nas-wang/nas-ssd/Results/RMP/log/PL.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
init_fancy_logging(log_file = log_file)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

hyperopt_search_dict = {
        "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear'},
        "pred_len": {6, 12, 18, 24}
    }

if __name__ == "__main__":
    #### Setp 1: Load configs
    args_parser = Benchmark_args()
    args = args_parser.parse_args()
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    #### Step 2: hyper-parameters optimaztion. 遍历models和pred_len.
    args.project = 'PL'
    args.model = 'LSTM'  # 'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'
    args.is_hyperopt = True
    args.is_save_checkpoint = False
    if args.is_hyperopt:
        logger.info(f"Starting hyperopt searching")
        logger.info(f"Working space {os.path.join(args.checkpoints, args.project)}")
        #for args.model in hyperopt_search_dict['model']:
        for pred_len in hyperopt_search_dict['pred_len']:
            args.pred_len= int(pred_len)
            settings = '{}_pl{}'.format(
                args.model,
                args.pred_len)
            hyperopt = Hyperoptimizer(args,settings)
            result_dir = os.path.join(args.checkpoints, args.project, settings)
            logger.info(f"-------------------------------------")
            logger.info(f"Current model and prediction length {settings}")
            max_trials = 50 
            args.patience = 10
            max_epochs = 50 # except XGBoostTSF
            # max_epochs = args.train_epochs # except XGBoostTSF
            bets_para = hyperopt.run_trials(result_dir,max_trials,max_epochs)
            logger.info(f"Best hyper-parameters: {bets_para}")
            logger.info(f"-------------------------------------")
    args.is_hyperopt = False
    ### Step 3: Training models with the optimized hyper parameters. Reading each parameters from saved hyperopt files.



    args.project = 'PL'
    args.patience = 20
    args.train_epochs = 200
    args.is_save_checkpoint = True
    for model in hyperopt_search_dict['model']:
        for pred_len in hyperopt_search_dict['pred_len']:
            args.pred_len= int(pred_len)
            settings = '{}_pl{}'.format(
                            args.model,
                            args.pred_len)
            result_dir = os.path.join(args.checkpoints, args.project, settings)
            hyperopt = Hyperoptimizer(args,settings)
            best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
            hyper_para = best_trial['result']['applied_hyper_paras']
            current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
            use_wandb = False
            try:
                wandb.init(
                    config=hyper_para,
                    project=f"{args.project}",
                    dir= os.path.join(args.checkpoints, args.project, settings),
                    anonymous="allow",
                    name=f"{args.project}_{settings}",
                )
                use_wandb = True
            except wandb.errors.Error:
                logger.warning(
                    "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
                )
            pop_exp = Exp_Pop_Main(current_args)
            trained_model, vali_loss, training_cost_time = pop_exp.train(settings)
            rmse_mean, re_rmse_mean, avg_inference_time, model_params, macs, params = pop_exp.test(settings, model=trained_model)
            result_json_dir = os.path.join(args.checkpoints, args.project, "result.json")
            result_data = {
                "project": args.project,
                "settings": settings,
                "hyper_para":hyper_para,
                "rmse_mean": rmse_mean,
                "re_rmse_mean": re_rmse_mean,
                "training_cost_time":training_cost_time,
                "avg_inference_time": avg_inference_time,
                "model_params": model_params,
                "macs": macs,
                "params":params
            }
            try:
                with open(result_json_dir, 'r') as f:
                    all_results = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_results = {"experiments": []}  

            all_results["experiments"].append(result_data)

            with open(result_json_dir, 'w') as f:
                json.dump(all_results, f, indent=4)

            wandb.log({
            "rmse_mean": rmse_mean,
            "re_rmse_mean":re_rmse_mean,
            "inference_time": avg_inference_time,
            "vali_loss":vali_loss,
            "training_cost_time":training_cost_time,}, commit=True)
            logger.info(f"-------------------End testing-------------------------")
            print(f"-------------------End testing-------------------------")
            
            if use_wandb:
                wandb.finish()

