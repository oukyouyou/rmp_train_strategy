import os, multiprocessing
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count())
from exp.exp_main import Exp_Basic
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop
from utils.metrics import metric
from resp_db.data_loader import Create_dataset_population, Create_dataset_individual
from utils.logger import LoggerMixin
import wandb
from main_configs import MODEL_DICT
import xgboost as xgb
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import pandas as pd
import time
import logging
from sklearn.metrics import mean_squared_error
import warnings
import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict
from models.XGBoostTSF import Model


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True) 

class Exp_xgboost():
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
# class Exp_xgboost(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_xgboost, self).__init__(args)
#         self.args = args

    def _build_model(self):
        model = MODEL_DICT[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_pop_data(self):
        train_loader, val_loader = Create_dataset_population(self.args)
        return train_loader, val_loader

    def _get_ps_data(self):
        train_loader, val_loader, test_loader = Create_dataset_individual(self.args,self.args.patient_id , self.args.f_num)
        return train_loader, val_loader, test_loader


    def _get_test_data(self):
        dataloader_project = 'test'
        test_loader = Create_dataset_individual(self.args, project = dataloader_project)
        return test_loader
    
    def _load_from_dataloader(self,dataloader):
        X_list, y_list = [], []
        total_batches = len(dataloader)
        last_print_percent = 0

        for i, batch in enumerate(dataloader):
            percent = (i + 1) / total_batches * 100
            # if percent - last_print_percent >= 2 or (i + 1) == total_batches:
            #     print(f"Loading batches: {percent:.1f}% ({i+1}/{total_batches})")
            #     last_print_percent = percent
            X_batch = batch["series_x"].float()
            y_batch = batch["series_y"].float()
            X_flat = X_batch.numpy().squeeze(-1) 
            y_flat = y_batch.numpy().squeeze(-1) 
            y_flat = y_flat[:,-1:]
            X_list.append(X_flat)  
            y_list.append(y_flat)
        X_train = np.concatenate(X_list, axis=0)
        y_train = np.concatenate(y_list, axis=0)
        return X_train, y_train 

    def train(self,setting, trial=1 ,load_model_path = None):
        logger.info(f"-------------------Start fitting-------------------------")
        logger.info(f"Settings: {setting}")
        try:
            if load_model_path:
                print(f'Loading model from: {load_model_path}')
                self.model.load(load_model_path)
                logger.info(f"Loading model from: {load_model_path}")
                print("Model loaded successfully")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}") from e
        
        pid_settings = 'pid{}_fnum{}'.format(
                self.args.patient_id,
                self.args.f_num)
            
        if self.args.project == "PL":
            train_loader , vali_loader  = self._get_pop_data()
            path = os.path.join(self.args.checkpoints, self.args.project, setting)
        else:
            train_loader , vali_loader, _  = self._get_ps_data()
            
            path = os.path.join(
                self.args.checkpoints,
                self.args.project + ("_fine_tuning" if self.args.is_fine_tuning else ""),
                setting,
                pid_settings
            )
        
        if not os.path.exists(path):
            os.makedirs(path)

        X_train, y_train  = self._load_from_dataloader(train_loader) 
        X_val, y_val = self._load_from_dataloader(vali_loader)

        #print(f"SHAPES: {X_train.shape=}, {y_train.shape=}")
        print('Fitting')
        time_now = time.time()
        train_outputs = self.model.forward( X_train, y_train)
        training_time = time.time() - time_now
        train_outputs = train_outputs.reshape(-1)
        y_train = y_train.reshape(-1)
        train_loss = mean_squared_error(train_outputs,y_train)
        vali_outputs = self.model.predict_(X_val)
        vali_loss = mean_squared_error(vali_outputs,y_val)

        vali_outputs = vali_outputs.reshape(-1)
        y_val = y_val.reshape(-1)

        print('Train loss',train_loss)
        print('Validation loss',vali_loss)

        model_path = path + '/' + 'xgboost_model.json'
        model = self.model
        model.save(model_path)

        with open(os.path.join(path, 'loss_log.log'), 'a') as f:
            f.write("Trials\tTrain Loss\tVal Loss\n")
            f.write(f"{trial}\t{train_loss:.6f}\t{vali_loss:.6f}\n")

        logger.info(f"-------------------End fitting-------------------------")
        return self.model, vali_loss , training_time
    
    def test(self, setting, model = None, load_model_path = None):
        logger.info(f"-------------------Start testing-------------------------")
        if self.args.project == "PL":
            test_loader = self._get_test_data()
        else:
            _ , _, test_loader  = self._get_ps_data()

        if model is not None:
            self.model = model
        else: ##### Remained to fix
            try:
                if load_model_path:
                    print(f'Loading model from: {load_model_path}')
                    self.model.load(default_path)
                    logger.info(f"Loading model from: {load_model_path}")
                else:
                    default_path = os.path.join(self.args.checkpoints, self.args.project, setting, 'xgboost_model.json')
                    print(f'Loading model from default path: {default_path}')

                    #self.model = torch.load(default_path, map_location=self.device)
                    #self._load_new_hyper_para(setting) 
                    self.model.load(default_path)

                    logger.info(f"Loading model from optimzed: {default_path}")
                print("Model loaded successfully")
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found: {e.filename}") from e
            except Exception as e:
                raise RuntimeError(f"Error loading model: {str(e)}") from e

        preds_by_patient = defaultdict(list)
        trues_by_patient = defaultdict(list)
        no_preds_by_patient = defaultdict(list)
        patient_ids = set()
        timings = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader),total=len(test_loader), 
                                desc="Testing progress", 
                                ncols=100,):

                X_batch = batch["series_x"].float().cpu()
                y_batch = batch["series_y"].float().cpu()
                X_flat = X_batch.numpy().squeeze(-1) 
                y_flat = y_batch.numpy().squeeze(-1) 
                y_flat = y_flat[:,-1:]
                patient_id = batch["name"][0]
                start_time = time.perf_counter()
                outputs = self.model.predict_(X_flat)
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)
                
                pred = outputs
                true = y_flat
                no_pred = X_flat[:,-1:]

                preds_by_patient[patient_id].append(pred)
                trues_by_patient[patient_id].append(true)
                no_preds_by_patient[patient_id].append(no_pred)
                patient_ids.add(patient_id)

        avg_time = sum(timings) / len(timings)
        std_time = torch.std(torch.tensor(timings)).item()
        # model_params = self._count_xgboost_params(model.get_booster())
        # macs = self._estimate_xgboost_macs(model.get_booster(), num_features=int(self.args.seq_len))

        # logger.info('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
        # logger.info('{:<30}  {:<8}'.format('Computational complexity: ', f"{macs:.1f} MACs/样本"))
        logger.info(f'Average inference time {avg_time}')
        logger.info(f'Standard devation inference time {std_time}')

        # result save

        test_result_folder_path = os.path.join(
                self.args.checkpoints,
                self.args.project + ("_fine_tuning" if self.args.is_fine_tuning else ""),
                setting,
                'test_results'
            )
        os.makedirs(test_result_folder_path, exist_ok=True)
        patient_ids = list(patient_ids) 
        suffix = "_fine_tuning" if self.args.is_fine_tuning else ""
        result_path = os.path.join(test_result_folder_path, f"result{suffix}.txt")

        with open(result_path, 'a') as f:
            #f.write(f"{setting}\n\n")
            rmse_list = []
            re_rmse_list = []
            for patient in patient_ids:
                preds = np.stack(preds_by_patient[patient])
                trues = np.stack(trues_by_patient[patient])
                no_preds = np.stack(no_preds_by_patient[patient])

                no_preds = no_preds.squeeze()[None, :]
                trues = trues.squeeze()[None, :]
                preds = preds.squeeze()[None, :]

                np.save(os.path.join(test_result_folder_path, f"{patient}_pred.npy"), preds)
                np.save(os.path.join(test_result_folder_path, f"{patient}_true.npy"), trues)
                np.save(os.path.join(test_result_folder_path, f"{patient}_x.npy"), no_preds)

                mae, mse, rmse, mape, mspe, rse, corr, relative_rmse = metric(preds, trues, no_preds)
                rmse_list.append(rmse)
                re_rmse_list.append(relative_rmse)
                tqdm.write(f"Patient: {patient}, mse: {mse:.7f}, rmse: {rmse:.7f}, relative_rmse: {relative_rmse:.7f}")
                f.write(f"Patient: {patient}, mse: {mse:.7f}, mae: {mae:.7f}, rmse: {rmse:.7f}, "
                        f"corr: {corr:.7f}, relative_rmse: {relative_rmse:.7f}\n\n")
                

                no_preds = no_preds.squeeze()
                trues = trues.squeeze()
                preds = preds.squeeze()
                index = np.arange(len(preds)) / self.args.sampling_rate_hz
                prediction_horizon_ms = int(self.args.pred_len * 1000 / self.args.sampling_rate_hz)

                plt.figure(figsize=(18, 6))
                plt.plot(index, no_preds, label="Non prediction", linewidth=2)
                plt.plot(index, trues, label="Ground truth", linewidth=2)
                plt.plot(index, preds, label="Predicted", linewidth=2, linestyle='--')
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.title(f"Breathing curves and prediction results for {self.args.model} - {prediction_horizon_ms}-ms prediction horizon")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.savefig(os.path.join(test_result_folder_path, f"{patient}_plot.png"))
                plt.close()

                # wandb.log({
                #    "pid": patient,
                #    "rmse": rmse,
                #    "re_rmse":relative_rmse,})

            if self.args.project == "PL":
                rmse_mean = np.mean(rmse_list)
                rmse_std = np.std(rmse_list)
                rmse_median = np.median(rmse_list)
                rmse_q1, rmse_q3 = np.percentile(rmse_list, [25, 75])

                re_rmse_mean = np.mean(re_rmse_list)
                re_rmse_std = np.std(re_rmse_list)
                re_rmse_median = np.median(re_rmse_list)
                re_rmse_q1, re_rmse_q3 = np.percentile(re_rmse_list, [25, 75])

                print("\n=== Inference time Statistics ===")
                print(f'Average inference time {avg_time}')
                print(f'Standard devation inference time {std_time}')

                print("\n=== Patient-wise RMSE Statistics ===")
                print(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}")
                print(f"Median: {rmse_median:.7f}")
                print(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]")

                print("\n=== Patient-wise Relative RMSE Statistics ===")
                print(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}")
                print(f"Median: {re_rmse_median:.7f}")
                print(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]")
                
                f.write("\n=== Inference time Statistics ===\n")
                f.write(f"Average: {avg_time:.7f} ms\n")  
                f.write(f"Std Dev: {std_time:.7f} ms\n")
                # f.write(f'Trainable parameter count: {model_params/1e6:.2f}M\n')
                # f.write(f"{'Computational complexity:':<30} {macs}\n")

                f.write("\n=== Patient-wise RMSE Statistics ===\n")
                f.write(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n")
                f.write(f"Median: {rmse_median:.7f}\n")
                f.write(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]\n\n")

                f.write("=== Patient-wise Relative RMSE Statistics ===\n")
                f.write(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n")
                f.write(f"Median: {re_rmse_median:.7f}\n")
                f.write(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]\n\n")

                rmse_path = os.path.join(test_result_folder_path, "rmse_list.npy")
                re_rmse_path = os.path.join(test_result_folder_path, "re_rmse_list.npy")
                np.save(rmse_path, np.array(rmse_list))
                np.save(re_rmse_path, np.array(re_rmse_list))
                
            else:
                logger.info(f"-------------------End testing-------------------------")
                return patient_id, rmse , relative_rmse, avg_time

    def _count_xgboost_params(self,booster):
        total_params = 0
        trees = booster.get_dump(with_stats=True)
        
        for tree in trees:
            nodes = tree.split('\\n')
            total_params += len([n for n in nodes if 'leaf' not in n]) * 3  
            total_params += len([n for n in nodes if 'leaf' in n])         
        
        return total_params
    
    def _estimate_xgboost_macs(self, booster, num_features):
        trees = booster.get_dump()
        total_macs = 0
        
        for tree in trees:
            depth = max([len(line.split('[')) for line in tree.split('\\n') if 'leaf' not in line])
            total_macs += depth * num_features  
        return total_macs
