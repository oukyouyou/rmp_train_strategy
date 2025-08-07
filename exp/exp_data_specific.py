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

class Exp_DS_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_DS_Main, self).__init__(args)
        self.args = args
        if hasattr(self.args, 'pretrained_path') and self.args.pretrained_path: 
            self._load_pretrained_model(self.args.pretrained_path)

    def _load_pretrained_model(self, pretrained_path: str):
        try:
            if os.path.exists(pretrained_path):  
                pretrained_dict = torch.load(pretrained_path, map_location=self.device)  
                if isinstance(self.model, nn.DataParallel):
                    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
                self.model.load_state_dict(pretrained_dict, strict=False)  # In case pre-trained model with diff para.
                print(f"Loaded pretrained model from: {pretrained_path}")
            else:
                print(f"Pretrained path not found, training from scratch: {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}. Training from scratch.")

    def _build_model(self):
        model = MODEL_DICT[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        train_loader, val_loader, test_loader = Create_dataset_individual(self.args,self.args.patient_id , self.args.f_num)
        return train_loader, val_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self, setting, load_model_path=None):
        logger.info(f"-------------------Start training on patient-specific dataset-------------------------")
        logger.info(f"Settings: {setting}")
        try:
            if load_model_path:
                print(f'Loading pre-trained model from: {load_model_path}')
                self.model = torch.load(load_model_path, map_location=self.device)
                logger.info(f"Loading pre-trained model from: {load_model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {e.filename}") from e
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}") from e
        
        train_loader , vali_loader, test_loader  = self._get_data()

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

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=False, is_save_checkpoint = self.args.is_save_checkpoint)
        train_log = {
            'epochs': [],    
            'iters': [],      
            'loss': [],      
            'val_loss': [],
            'speed': [],      
            'cost_time': []   
        }
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        time_now = time.time()
        time_start = time.time()
        time_now_epoch = time.time()

        best_vali_loss = 1
        vali_loss = 1
        training_end_time = None
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, batch in enumerate(train_loader):
                # batch = (name, time_x, batch_series_x, batch_series_noisy_x, time_y, batch_series_y, batch_series_noisy_y)
                patient_id = batch["name"]
                #print(f'======Train==========={patient_id}==================')
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float().to(self.device, non_blocking=True)
                #print('--------Shape of model input:--------',batch_x.shape)
                #print('--------Shape of model target:--------',batch_y.shape)

                if self.args.use_amp:
                    with torch.amp.autocast():
                        if 'former' in self.args.model:
                            outputs = self.model(batch_x)
                            outputs = outputs.unsqueeze(-1)  # [batch, out_seq_len, 1]  
                        else:
                            outputs = self.model(batch_x)
                        #print('--------Shape of model output:--------',outputs.shape)
                        outputs = outputs[:, -self.args.pred_len:, -1:]
                        batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device, non_blocking=True)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'former' in self.args.model:
                        outputs = self.model(batch_x)
                        outputs = outputs.unsqueeze(-1)  # [batch, out_seq_len, 1]  
                    else:
                        outputs = self.model(batch_x)
                
                    #print('--------Shape of model output:--------',outputs.shape)
                    if self.args.MO_flag:
                        outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                    else:
                        outputs = outputs[:, -1:, -self.args.output_dim:]
                        batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)                      
                    #print('--------Shape of model output after -1:--------',outputs.shape)

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    #print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    cost_time = time.time() - time_now
                    #print('\tspeed: {:.8f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    train_log['epochs'].append(epoch + 1)
                    train_log['iters'].append(i + 1)
                    train_log['loss'].append(loss.item())
                    train_log['val_loss'].append(vali_loss)
                    train_log['speed'].append(speed)
                    train_log['cost_time'].append(cost_time)
                    iter_count = 0
                    #time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            #print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            if best_vali_loss is None:
                best_vali_loss = vali_loss
                training_end_time = time.time()

            elif best_vali_loss > vali_loss:
                best_vali_loss = vali_loss
                training_end_time = time.time()
                
            cost_time_epoch = time.time() - time_now_epoch
            # print(f"Cost time (s) in each epoch: {cost_time_epoch}")
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(epoch + 1, train_steps, train_loss, vali_loss))
            
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(path, f'checkpoint_epoch{epoch + 1}_tran_loss{train_loss}_val_loss{vali_loss}.pth')
                #torch.save(self.model, checkpoint_path)
                #print(f"Saved checkpoint at epoch {epoch + 1}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                logger.info(f"Early stopped at epoch {epoch} Best vali loss {best_vali_loss}")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        if training_end_time is None:
            training_end_time = time.time()
        training_time = training_end_time - time_start
        print(f"Training cost time: {training_time:.2f}s")
        logger.info(f"Training cost time: {training_time:.2f}s")
        
        if self.args.is_save_checkpoint:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model = torch.load(best_model_path)
            logger.info(f"Loaded best checkpoint")

        df = pd.DataFrame(train_log)
        csv_path = os.path.join(path, f'train_log.csv') 
        df.to_csv(csv_path, index=False)

        logger.info(f"-------------------End training-------------------------")
        return self.model, best_vali_loss ,training_time
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float()

                time_now = time.time() 
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if 'former' in self.args.model:
                            outputs = self.model(batch_x)
                            outputs = outputs.unsqueeze(-1)  # [batch, out_seq_len, 1]  
                        else:
                            outputs = self.model(batch_x)
                else:
                    if 'former' in self.args.model:
                        outputs = self.model(batch_x)
                        outputs = outputs.unsqueeze(-1)  # [batch, out_seq_len, 1]  
                    else:
                        outputs = self.model(batch_x)
                speed = (time.time() - time_now)
                #print('\tspeed: {:.8f}s/iter'.format(speed))
                if self.args.MO_flag:
                    outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                else:
                    outputs = outputs[:, -1:, -self.args.output_dim:]
                    batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)   

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, model=None, load_model_path = None):
        logger.info(f"-------------------Start testing-------------------------")
        _ , _, test_loader  = self._get_data()
        pid_settings = 'pid{}_fnum{}'.format(
                self.args.patient_id,
                self.args.f_num)
        
        if model is not None:
            self.model = model
        else:
            try:
                if load_model_path:
                    print(f'Loading model from: {load_model_path}')
                    self.model = torch.load(load_model_path, map_location=self.device)
                    logger.info(f"Loading model from: {load_model_path}")
                else:
                    default_path = os.path.join(self.args.checkpoints, self.args.project, setting, pid_settings, 'checkpoint.pth')
                    print(f'Loading model from default path: {default_path}')
                    self.model = torch.load(default_path, map_location=self.device)
                    self.model.to(self.device)

                    logger.info(f"Loading model from optimzed: {default_path}")
                print("Model loaded successfully")
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found: {e.filename}") from e
            except Exception as e:
                raise RuntimeError(f"Error loading model: {str(e)}") from e
            
        preds_by_patient = []
        trues_by_patient = []
        no_preds_by_patient = []
        timings = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float().to(self.device, non_blocking=True)
                patient_id = batch["name"][0]
                # print(f'======Test==========={patient_id}==================')

                start_time = time.perf_counter()
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if 'former' in self.args.model:
                            outputs = self.model(batch_x)
                            outputs = outputs.unsqueeze(-1)
                        else:
                            outputs = self.model(batch_x)
                else:
                    if 'former' in self.args.model:
                        outputs = self.model(batch_x)
                        outputs = outputs.unsqueeze(-1)
                    else:
                        outputs = self.model(batch_x)
                torch.cuda.synchronize()  
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)

                outputs = outputs[:, -1:, -1:].view(1)
                batch_y = batch_y[:, -1:, -1:].view(1) # .to(self.device, non_blocking=True)
                no_pred = batch_x[:, -1:, -1:].view(1)
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                no_pred = no_pred.detach().cpu().numpy()
                preds_by_patient.append(pred)
                trues_by_patient.append(true)
                no_preds_by_patient.append(no_pred)

        avg_inference_time = sum(timings) / len(timings)
        std_inference_time = torch.std(torch.tensor(timings)).item()       
        if self.args.test_flop:
            model_params, macs, params = test_params_flop(self.model,(batch_x.shape[1],batch_x.shape[2]))
            logger.info('INFO: Trainable parameter count: {:.5f}M'.format(model_params / 1000000.0))
            logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
        logger.info(f'Average inference time {avg_inference_time}')
        logger.info(f'Standard devation inference time {std_inference_time}')   
        
        # result save
        folder_path = os.path.join(
                self.args.checkpoints,
                self.args.project + ("_fine_tuning" if self.args.is_fine_tuning else ""),
                setting,
                 'test_results'
            )
        os.makedirs(folder_path, exist_ok=True)

        suffix = "_fine_tuning" if self.args.is_fine_tuning else ""
        result_path = os.path.join(folder_path, f"result{suffix}.txt")
        with open(result_path, 'a') as f:
            #f.write(f"{setting}\n\n")

            preds = np.stack(preds_by_patient)
            trues = np.stack(trues_by_patient)
            no_preds = np.stack(no_preds_by_patient)
            np.save(os.path.join(folder_path, f"{patient_id}_pred.npy"), preds)
            np.save(os.path.join(folder_path, f"{patient_id}_true.npy"), trues)
            np.save(os.path.join(folder_path, f"{patient_id}_x.npy"), no_preds)

            mae, mse, rmse, mape, mspe, rse, corr, relative_rmse = metric(preds, trues, no_preds)

            print(f"Patient: {patient_id}, mse: {mse:.7f}, rmse: {rmse:.7f}, relative_rmse: {relative_rmse:.7f}")
            f.write(f"Patient: {patient_id}, mse: {mse:.7f}, mae: {mae:.7f}, rmse: {rmse:.7f}, "
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
        return patient_id, rmse , relative_rmse, avg_inference_time
    





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
    