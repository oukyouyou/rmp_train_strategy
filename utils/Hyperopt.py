import datetime
from torch.utils.data import Dataset
import torch.nn as nn
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, tpe
from copy import deepcopy
from pathlib import Path
import dill
from hyperopt import hp
import time
import os
from argparse import Namespace
import logging
import wandb
import json

from utils.search_spaces import SEARCH_SPACE_LINEAR, SEARCH_SPACE_LSTM, SEARCH_SPACE_MLP, SEARCH_SPACE_TRANSFORMER, SEARCH_SPACE_TRANSFORMER_TSFv2, SEARCH_SPACE_DLINEAR, SEARCH_SPACE_XGB, SEARCH_SPACE_ARIMA, SEARCH_SPACE_LSTM_batch_1, SEARCH_SPACE_MAMBA,SEARCH_SPACE_TIMESNET
from resp_db.client import RpmDatabaseClient
from exp.exp_pop_pre_train import Exp_Pop_Main
from exp.exp_patient_specific import Exp_PS_Main
from exp.exp_data_specific import Exp_DS_Main
from exp.exp_xgboost import Exp_xgboost
from exp.exp_statistical_model import Exp_SM_Main

from utils.logger import LoggerMixin, init_fancy_logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Hyperoptimizer(LoggerMixin):
    def __init__(self, args: dict, settings: str):
        self.args = args
        self.model = args.model
        self.start_time = time.time() 
        self.start_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.settings = settings
        #self.trials = None
        self.trials = Trials() 
        self.training_cost_time = 0

    def _get_space(self, model):
        if model == "MLP":
            return SEARCH_SPACE_MLP
        elif model == "LSTM":
            if self.args.project == 'PL':
                return SEARCH_SPACE_LSTM
            else:
                return SEARCH_SPACE_LSTM_batch_1
        elif model == "TransformerTSFv2":
            return SEARCH_SPACE_TRANSFORMER_TSFv2
        elif model == "DLinear":
            return SEARCH_SPACE_DLINEAR
        elif model == "XGBoostTSF":
            return SEARCH_SPACE_XGB
        elif model == "Arima":
            return SEARCH_SPACE_ARIMA
        elif model == "Mamba":
            return SEARCH_SPACE_MAMBA
        elif model == "TimesNet":
            return SEARCH_SPACE_TIMESNET

        else:
            print('Errors in model name in _get_space()')
            
    @staticmethod
    def _merge_hyperparameters(args: Namespace, hyper_para: dict) -> Namespace:
        INT_PARAMS = {'seq_len', 'max_depth', 'min_child_weight', 'n_estimators'} 
        # current_args = deepcopy(args)
        # merged_args = vars(args).copy()
        # merged_args.update(hyper_para)
        # current_args = Namespace(**merged_args)
        # return current_args
        merged_args = vars(args).copy()
        for k, v in hyper_para.items():
            if k in INT_PARAMS:
                merged_args[k] = int(round(v)) 
            else:
                merged_args[k] = v
        return Namespace(**merged_args)

    def _predictor(self, hyper_para):
        current_args = self._merge_hyperparameters(self.args, hyper_para)
        logger.info(f"Optimizing for current settings: {self.settings}")
        logger.info("Testing current hyper parameters:")
        for key, value in hyper_para.items():
            logger.info(f"  {key}: {value}")
        
        current_trial_number = len(self.trials) 
        print(f"Running trial {current_trial_number }")
        # logger.info(f"Running trial {current_trial_number}")
        use_wandb = False
        if self.args.project == 'PL':
            try:
                wandb.init(
                    config=hyper_para,
                    project=f"{self.args.project}",
                    dir= os.path.join(self.args.checkpoints, self.args.project, self.settings),
                    anonymous="allow",
                    name=f"Hyperopt_{self.settings}_{current_trial_number}",
                )
                use_wandb = True
            except wandb.errors.Error:
                logger.warning(
                    "Cannot import weights and biases. Consider creating a free account and login for advanced results tracking!"
                )
        if self.args.project == 'PL':
            if not self.model == 'XGBoostTSF':
                pop_exp = Exp_Pop_Main(current_args)
                _, vali_loss, self.training_cost_time = pop_exp.train(self.settings)
            else:
                pop_exp = Exp_xgboost(current_args)
                _, vali_loss, self.training_cost_time = pop_exp.train(self.settings, trial = current_trial_number)
        
        elif self.args.project == 'PS-4DCT' or self.args.project == 'PS':
            if not self.model == 'XGBoostTSF':
                ps_exp = Exp_PS_Main(current_args)
                _, vali_loss, self.training_cost_time = ps_exp.train(self.settings)
            else:
                pop_exp = Exp_xgboost(current_args)
                _, vali_loss, self.training_cost_time = pop_exp.train(self.settings, trial = current_trial_number)
        
        elif self.args.project == 'DS':
            if self.model == 'XGBoostTSF':
                ds_exp = Exp_xgboost(current_args)
                _, vali_loss, self.training_cost_time = ds_exp.train(self.settings, trial = current_trial_number)
            elif self.model in ['Arima', 'wLMS']:
                ds_exp = Exp_SM_Main(current_args)
                _, vali_loss, self.training_cost_time = ds_exp.predict(self.settings, flag = 'train')
            else:
                ds_exp = Exp_DS_Main(current_args)
                _, vali_loss, self.training_cost_time = ds_exp.train(self.settings)

        if use_wandb:
            wandb.finish()
        results = {
            "loss": vali_loss,
            "status": STATUS_OK,
            "applied_hyper_paras": hyper_para,
            "settings": self.settings,
            "training_cost_time": self.training_cost_time,
        }
        return results
    
    def run_trials(self, result_dir: Path, max_trials: int, max_epochs: int):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)  
        filepath = os.path.join(result_dir, self.settings + ".hyperopt")
        trials_step = 0
        self.args.train_epochs = max_epochs # Except XGBoostTSF
        logger.info(
            f"Local result directory of current hyper-optimizer run: {result_dir}"
        )
       
        try:  # try to load an already saved trials object, and increase the max
            trials = dill.load(open(filepath, "rb"))
            # remove trials with status fail
            index = len(trials.statuses()) - 1
            while index > -1:
                if trials.statuses()[index] == "fail":
                    trials.trials.pop(index)
                index -= 1
            max_trials = len(trials.trials) + trials_step
        except FileNotFoundError:
            # initialize empty trials database
            trials = Trials()
            self.trials = trials
            logger.info(
            f"Optimizing from scratch: {result_dir}"
            )
        
        best = fmin(
            fn=self._predictor,
            space=self._get_space(self.model),
            algo=tpe.suggest,
            max_evals=max_trials,
            trials=trials,
            return_argmin=False,
            show_progressbar=False,
        )
        end_time = time.time()
        total_time_seconds = end_time - self.start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time_seconds))) 
        print(f"Settings: {self.settings}")
        logger.info(f"Optimizing for current settings: {self.settings}")
        print(f"Best hyper-parameters so far: {best}")
        logger.info(f"Best hyper-parameters so far: {best}")
        print(f"Optimization at: {self.start_datetime}")
        logger.info(f"Optimization at: {self.start_datetime}")
        print(f"Total time consumed: {total_time_str}")
        logger.info(f"Total time consumed: {total_time_str}")
        
        # save the trials object
        with open(filepath, "wb") as f:
            print(f"Model arch was saved at {filepath}")
            logger.info(f"ModelArch was saved at {filepath}")
            dill.dump(trials, f)
            
        #self._save_trials(trials,result_dir)
        return best
    
    @staticmethod
    def read_hyperopt_object(filepath: str, settings: str, json_filename="trial_results.json") -> hyperopt.Trials:
        hyperopt_filepath = os.path.join(filepath, settings+ ".hyperopt")
        with open(hyperopt_filepath, "rb") as f:
            trials = dill.load(f)
        best_trial = trials.best_trial
        summary = (
            "Found saved Trials! \n "
            "------------- Hyper para optimzed information --------------------\n"
            f"Setting name: {best_trial['result']['settings']} \n"
            f"Number of evaluated combinations: {len(trials.trials)} \n"
            f"Best MSE loss on val set: {best_trial['result']['loss']} \n "
            f"Corresponding hyperparas: {best_trial['result']['applied_hyper_paras']} \n"
            "------------- End information ----------------"
        )
        save_para = {
            "settings": best_trial['result']['settings'],
            "vali loss": best_trial['result']['loss'] ,
            "hyper_para": best_trial['result']['applied_hyper_paras']
        }
        print(summary)
        #logger.info(summary)
        json_path = os.path.join(filepath, json_filename)
        with open(json_path, "w") as f:
            json.dump(save_para, f, indent=4) 
        return best_trial

    