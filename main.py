import os
import sys
sys.argv = [sys.argv[0]] 
import random
import torch
import argparse
import numpy as np
from pathlib import Path
import logging
import wandb
from tqdm import tqdm

from resp_db.client import RpmDatabaseClient
from resp_db.new_dl_dataset_patient_wise import DeepLearningDatasetBuilder
from utils.logger import LoggerMixin, init_fancy_logging
# from exp.exp_pop_pre_train import Exp_Pop_Main
# from exp.exp_patient_specific import Exp_PS_Main
# from exp.exp_data_specific import Exp_DS_Main
# from exp.exp_xgboost import Exp_xgboost
# from exp.exp_test import Exp_Test
from main_configs import Benchmark_args

# init_fancy_logging(log_file = "/mnt/nas-wang/nas-ssd/Results/RMP_new/log/hyperopt.log")
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

hyperopt_search_dict = {
        "model": {'LSTM', 'MLP', 'TransformerTSFv2', 'DLinear','XGBoostTSF'},
        "project": {'population-level','PS-4DCT', 'PS', 'DS'},
        "pred_len": {6, 12, 18, 24}
    }

def get_pids(args):
    client = RpmDatabaseClient(db_filepath=args.db_root)
    with client:
            raw_pairs = client.get_pids_fraction_of_test_dataset(project='test')
            pairs = list(raw_pairs)
    return pairs


if __name__ == "__main__":
    #### Setp 1: Load configs
    args_parser = Benchmark_args()
    args = args_parser.parse_args()
    fix_seed = args.random_seed
    #args.db_root = db_root
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    #### Step 2: Create new dataset.
    is_create_dataset =False
    if is_create_dataset:
        table_name = 'DeepLearningDataset'
        datas_builder = DeepLearningDatasetBuilder(args,table_name)
        args.table_name = datas_builder.dataset

