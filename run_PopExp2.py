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
        "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear', 'Mamba','TimesNet'},
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
    args.model = 'MLP'  # 'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'
    args.use_amp = False
    args.pred_len= int(12)
    args.is_hyperopt = True
    args.is_save_checkpoint = False
    if args.is_hyperopt:
        logger.info(f"Starting hyperopt searching")
        logger.info(f"Working space {os.path.join(args.checkpoints, args.project)}")
        #for args.model in hyperopt_search_dict['model']:

        settings = '{}_pl{}'.format(
            args.model,
            args.pred_len)
        hyperopt = Hyperoptimizer(args,settings)
        result_dir = os.path.join(args.checkpoints, args.project, settings)
        logger.info(f"-------------------------------------")
        logger.info(f"Current model and prediction length {settings}")
        max_trials = 50 
        args.patience = 5 # 10
        max_epochs = 10 # 50 # except XGBoostTSF
        # max_epochs = args.train_epochs # except XGBoostTSF
        bets_para = hyperopt.run_trials(result_dir,max_trials,max_epochs)
        logger.info(f"Best hyper-parameters: {bets_para}")
        logger.info(f"-------------------------------------")
    args.is_hyperopt = False
    