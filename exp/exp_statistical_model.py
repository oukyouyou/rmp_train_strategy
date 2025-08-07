import os, multiprocessing
from exp.exp_main import Exp_Basic
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop
from utils.metrics import metric
from resp_db.data_loader import Create_dataset_population, Create_dataset_individual
from utils.logger import LoggerMixin
from main_configs import MODEL_DICT
from collections import defaultdict
from tqdm import tqdm
import logging
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
import os
import time
from resp_db.client import RpmDatabaseClient
import wandb
import warnings
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings('ignore')

class Exp_SM_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_SM_Main, self).__init__(args)
        self.args = args
        if hasattr(self.args, 'pretrained_path') and self.args.pretrained_path: 
            self._load_pretrained_model(self.args.pretrained_path)

    def _build_model(self):
        model = MODEL_DICT[self.args.model].Model(self.args).float()
        #model = MODEL_DICT[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def MSE(self, pred, true):
        return np.mean((pred - true) ** 2)

    def RMSE(self,pred, true):
        return np.sqrt(self.MSE(pred, true))

    def Relat_RMSE(self, pred, true,no_pred):
        return float(self.RMSE(pred, true) / self.RMSE(no_pred, true))

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _get_data(self):
        train_loader, val_loader, test_loader = Create_dataset_individual(self.args,self.args.patient_id , self.args.f_num)
        return train_loader, val_loader, test_loader

    def predict(self, setting, flag = 'train'):
        train_loader , vali_loader, test_loader  = self._get_data()
        if flag == 'train':
            data_loader = train_loader
        else:
            data_loader = test_loader
        
        pid_settings = 'pid{}_fnum{}'.format(
                self.args.patient_id,
                self.args.f_num)
        path = os.path.join(
                self.args.checkpoints,
                self.args.project + ("_fine_tuning" if self.args.is_fine_tuning else ""),
                setting,
                pid_settings
            )
        os.makedirs(path, exist_ok=True)

        preds_by_patient = []
        trues_by_patient = []
        no_preds_by_patient = []
        criterion = self._select_criterion()
        timings = []
        pred_loss = []
        time_start = time.time()
        for i, batch in enumerate(data_loader):
            # print('Batch:',i)
            patient_id = batch["name"]
            batch_x = batch["series_x"].float()
            batch_y = batch["series_y"].float()
            #batch_x = batch_x.cpu().numpy()
            #batch_y = batch_y.cpu().numpy()

            start_time = time.perf_counter()
            # print('Shape:',batch_x.shape)
            outputs = self.model.forward(batch_x)
            end_time = time.perf_counter()
            timings.append((end_time - start_time) * 1000)
            if self.args.MO_flag:
                outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:]
                batch_x = batch_x[:, -1:, -self.args.output_dim:]   
            else:
                outputs = outputs[:, -1:, -self.args.output_dim:]
                batch_y = batch_y[:, -1:, -self.args.output_dim:] 
                batch_x = batch_x[:, -1:, -self.args.output_dim:]     

            # loss = self.MSE(outputs, batch_y)
            loss = criterion(outputs, batch_y)
            pred_loss.append(loss.item())
            # print(loss)
            
            # outputs = outputs[:, -1:, -1:].unsqueeze(0)
            # batch_y = batch_y[:, -1:, -1:].view(1) # .to(self.device)
            # no_pred = batch_x[:, -1:, -1:].view(1)
            pred = outputs.view(1)
            true = batch_y.view(1)
            no_pred = batch_x.view(1)

            # pred = pred.detach().cpu().numpy()
            # true = true.detach().cpu().numpy()
            # no_pred = no_pred.detach().cpu().numpy()
            
            print(pred)
            print(true)
            print(no_pred)
            print('Shape:',pred.shape)
            print('Shape:',true.shape)
            print('Shape:',no_pred.shape)

            preds_by_patient.append(pred)
            trues_by_patient.append(true)
            no_preds_by_patient.append(no_pred)

        pred_loss = np.average(pred_loss)
        training_time = time.time() - time_start

        if flag == 'test':
            folder_path = os.path.join(
                self.args.checkpoints,
                self.args.project,
                setting,
                 'test_results'
            )
            os.makedirs(folder_path, exist_ok=True)
            
            preds = np.stack(preds_by_patient)
            trues = np.stack(trues_by_patient)
            no_preds = np.stack(no_preds_by_patient)
            
            np.save(os.path.join(path, f"{patient_id}_pred.npy"), preds)
            np.save(os.path.join(path, f"{patient_id}_true.npy"), trues)
            np.save(os.path.join(path, f"{patient_id}_x.npy"), no_preds)

            rmse = self.RMSE(preds, trues)
            relative_rmse = self.Relat_RMSE(preds, trues,no_preds)
            
            avg_inference_time = sum(timings) / len(timings)
            std_inference_time = torch.std(torch.tensor(timings)).item()   

            result_path = os.path.join(folder_path, f"result.txt")
            with open(result_path, 'a') as f:
                #f.write(f"{setting}\n\n")
                f.write(f"Patient: {patient_id}, rmse: {rmse:.7f}, relative_rmse: {relative_rmse:.7f}\n")
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
                plt.savefig(os.path.join(folder_path, f"{patient_id}_plot.png"))
                plt.close()

                print("\n=== Inference time Statistics ===")
                print(f'Average inference time {avg_inference_time}')
                print(f'Standard devation inference time {std_inference_time}')

                print(f"\n=== Patient {patient_id} RMSE and relative RMSE ===")
                print(f"\nPatient RMSE mean: {rmse:.7f}")
                print(f"\nPatient relative RMSE mean: {relative_rmse:.7f}")

                f.write(f"\n=== Patient {patient_id} test results ===")
                f.write(f"Patient RMSE mean: {rmse:.7f}\n")
                f.write(f"Patient relative RMSE mean: {relative_rmse:.7f}\n")
                f.write(f"Inference time Average: {avg_inference_time:.7f} ms\n")  
                f.write(f"Inference time Std Dev: {std_inference_time:.7f} ms\n")
                # logger.info(f"\n=== Patient {patient_id} test results ===")
                # logger.info(f'Average Inference time {avg_inference_time}')
            return patient_id, rmse , relative_rmse, avg_inference_time, training_time, pred_loss
        else:
            return self.model, pred_loss, training_time
    





# class TrainingLogger(LoggerMixin):
#     def __init__(
#         self,
#         log_dir: str = "./logs",
#         log_file: str = "training_metrics.csv",
#         resume: bool = False,
#     ):
#         super().__init__()
#         self.log_dir = Path(log_dir)
#         self.log_file = self.log_dir / log_file
#         self.log_dir.mkdir(parents=True, exist_ok=True)
        
#         # Initialize CSV file (if not resuming)
#         if not resume or not self.log_file.exists():
#             with open(self.log_file, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["epoch", "train_loss", "val_loss", "speed", "lr"])
        
#         self.metrics: Dict[str, List[float]] = {
#             "epoch": [],
#             "train_loss": [],
#             "val_loss": [],
#             "speed": [],
#             "lr": [],
#         }

#     def log_epoch(
#         self,
#         epoch: int,
#         train_loss: float,
#         val_loss: Optional[float] = None,
#         speed: Optional[float] = None,
#         lr: Optional[float] = None,
#     ):
#         """Log metrics for a single epoch."""
#         self.metrics["epoch"].append(epoch)
#         self.metrics["train_loss"].append(train_loss)
#         self.metrics["val_loss"].append(val_loss)
#         self.metrics["speed"].append(speed)
#         self.metrics["lr"].append(lr)

#         # Log to console
#         self.logger.info(
#             f"Epoch {epoch:3d}: "
#             f"Train Loss = {train_loss:.4f}, "
#             f"Val Loss = {val_loss:.4f if val_loss is not None else 'N/A'}, "
#             f"Speed = {speed:.2f} samples/sec, "
#             f"LR = {lr:.6f if lr is not None else 'N/A'}"
#         )

#         # Append to CSV
#         with open(self.log_file, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([epoch, train_loss, val_loss, speed, lr])

#     def save_json(self, file_name: str = "metrics.json"):
#         """Save metrics to a JSON file."""
#         with open(self.log_dir / file_name, "w") as f:
#             json.dump(self.metrics, f, indent=4)

#     def get_metrics(self) -> Dict[str, List[float]]:
#         """Return logged metrics."""
#         return self.metrics
    