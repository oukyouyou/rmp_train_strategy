import os
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

from utils.Hyperopt import Hyperoptimizer
from utils.logger import LoggerMixin, init_fancy_logging
from exp.exp_pop_pre_train import Exp_Pop_Main
from exp.exp_patient_specific import Exp_PS_Main
from exp.exp_data_specific import Exp_DS_Main
from exp.exp_xgboost import Exp_xgboost
from exp.exp_test import Exp_Test
from main_configs import Benchmark_args

init_fancy_logging(log_file = "/mnt/nas-wang/nas-ssd/Results/RMP_new/log/hyperopt.log")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

hyperopt_search_dict = {
        "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'},
        "project": {'population-level','FT-4DCT', 'patient-specific', 'test'},
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

    # args.model = 'XGBoostTSF'
    # args.pred_len= int(12)
    # settings = '{}_pl{}'.format(
    #     args.model,
    #     args.pred_len)
    # hyperopt = Hyperoptimizer(args,settings)
    # result_dir = os.path.join(args.checkpoints, args.project, settings)
    # logger.info(f"-------------------------------------")
    # logger.info(f"Current model and prediction length {settings}")
    # max_trials = 50 
    # max_epochs = 3 # except XGBoostTSF
    # bets_para = hyperopt.run_trials(result_dir,max_trials,max_epochs)
    # logger.info(f"Best hyper-parameters: {bets_para}")
    # logger.info(f"-------------------------------------")

    #### Step 2: hyper-parameters optimaztion. 遍历models和pred_len.
    is_hyperopt = False
    if is_hyperopt:
        logger.info(f"Starting hyperopt searching")
        logger.info(f"Working space {os.path.join(args.checkpoints, args.project)}")
        for model in hyperopt_search_dict['model']:
            for pred_len in hyperopt_search_dict['pred_len']:
                args.model = model
                args.pred_len= int(pred_len)
                settings = '{}_pl{}'.format(
                    args.model,
                    args.pred_len)
                hyperopt = Hyperoptimizer(args,settings)
                result_dir = os.path.join(args.checkpoints, args.project, settings)
                logger.info(f"-------------------------------------")
                logger.info(f"Current model and prediction length {settings}")
                max_trials = 50 
                max_epochs = 3 # except XGBoostTSF
                bets_para = hyperopt.run_trials(result_dir,max_trials,max_epochs)
                logger.info(f"Best hyper-parameters: {bets_para}")
                logger.info(f"-------------------------------------")

    #### Step 3: Training models with the optimized hyper parameters. Reading each parameters from saved hyperopt files.
    # args.project = 'population-level'
    # args.model = 'LSTM' #'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'
    # args.pred_len = 24
    # args.train_epochs = 10
    # settings = '{}_pl{}'.format(
    #                 args.model,
    #                 args.pred_len)
    # result_dir = os.path.join(args.checkpoints, args.project, settings)
    # hyperopt = Hyperoptimizer(args,settings)
    # best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
    # hyper_para = best_trial['result']['applied_hyper_paras']
    # current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
    # use_wandb = False
    # try:
    #     wandb.init(
    #         config=hyper_para,
    #         project=f"{args.project}",
    #         dir= os.path.join(args.checkpoints, args.project, settings),
    #         anonymous="allow",
    #         name=f"{args.project}_{settings}",
    #     )
    #     use_wandb = True
    # except wandb.errors.Error:
    #     logger.warning(
    #         "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
    #     )

    # if args.project == 'population-level':
    #     if not args.model == 'XGBoostTSF':
    #         pop_exp = Exp_Pop_Main(current_args)
    #         trained_model, vali_loss = pop_exp.train(settings)
    #         pop_exp.test(settings, model=trained_model)
    #     else:
    #         pop_exp = Exp_xgboost(current_args)
    #         trained_model, vali_loss = pop_exp.train(settings)
    # elif args.project == 'FT-4DCT' or args.project == 'patient-specific':
    #     ps_exp = Exp_PS_Main(current_args)
    #     pids = ps_exp._get_pids()
    #     for patient_id, f_num in pids:
    #         trained_model, vali_loss = ps_exp.train(settings, patient_id , f_num)
    # elif args.project == 'data-specific':
    #     ds_exp = Exp_DS_Main(current_args)
    #     trained_model, vali_loss = ds_exp.train(settings)
    # if use_wandb:
    #     wandb.finish()

    ### Step 4: Testing 
    args.project = 'population-level'
    for model in hyperopt_search_dict['model']:
        for pred_len in hyperopt_search_dict['pred_len']:
            args.model = model
            args.pred_len= int(pred_len)
            settings = '{}_pl{}'.format(
                args.model,
                args.pred_len)
            print('Testing :',settings)
            hyperopt = Hyperoptimizer(args,settings)
            result_dir = os.path.join(args.checkpoints, args.project, settings)
            best_trial = hyperopt.read_hyperopt_object(result_dir, settings)
            hyper_para = best_trial['result']['applied_hyper_paras']
            current_args = hyperopt._merge_hyperparameters(args, hyper_para) # Apply optimzed parameters
            
            pop_exp = Exp_Pop_Main(current_args)
            pop_exp.test(settings)


    # XGBoostTSF
    # pop_exp = Exp_xgboost(current_args)
    # model, vali_loss = pop_exp.train(settings)
    # print(vali_loss)
    # pop_exp.test(settings, model=model)
    
