from exp.exp_main import Exp_Basic
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop, read_hyperopt_object
from utils.metrics import metric
from resp_db.data_loader import Create_dataset_population, Create_dataset_individual
from main_configs import MODEL_DICT

from utils.logger import LoggerMixin
import wandb
from argparse import Namespace

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
import logging
from tqdm import tqdm
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt

from utils.logger import LoggerMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
warnings.filterwarnings('ignore')

class Exp_Pop_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Pop_Main, self).__init__(args)
        self.args = args

    def _build_model(self):
        # model_dict = {
        #     'DLinear': DLinear,
        #     'MLP': MLP, # Pure 3 layer LSTM
        #     'XGBoostTSF': XGBoostTSF,
        #     'wLMS': wLMS,
        #     'WaveletLSTM': WaveletLSTM,
        #     'LGEANet': LGEANet,
        #     'LSTM': Lin, 
        #     'TransformerTSFv2':TransformerTSFv2,
        #     'Many2Many':Many2Many,
        # }
        model = MODEL_DICT[self.args.model].Model(self.args).float()

        # print(model_dict[self.args.model])
        # print("Model Parameters:", sum(p.numel() for p in model.parameters()))
        # print("State dict keys:", list(model.state_dict().keys()))
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        train_loader, val_loader = Create_dataset_population(self.args)
        return train_loader, val_loader
    
    def _get_test_data(self):
        dataloader_project = 'test'
        test_loader = Create_dataset_individual(self.args, project = dataloader_project)
        return test_loader
    
    def _load_new_hyper_para(self,settings):
        result_dir = os.path.join(self.args.checkpoints, self.args.project, settings)
        best_trial = read_hyperopt_object(result_dir, settings)
        hyper_para = best_trial['result']['applied_hyper_paras']
        current_args = self._merge_hyperparameters(self.args, hyper_para) # Apply optimzed parameters
        self.args = current_args
        self._build_model()
        logger.info(f"Loading new hyper para from {result_dir}")

    def _merge_hyperparameters(self, args: Namespace, hyper_para: dict) -> Namespace:
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
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self, setting):
        logger.info(f"-------------------Start training on population-based dataset-------------------------")
        logger.info(f"Settings: {setting}")
        train_loader , vali_loader  = self._get_data()
        path = os.path.join(self.args.checkpoints, self.args.project, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, is_save_checkpoint = self.args.is_save_checkpoint)
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
        time_now_epoch = time.time()
        time_start = time.time()
        best_vali_loss = 1
        vali_loss = 1
        for epoch in tqdm(range(self.args.train_epochs), 
                 total=self.args.train_epochs, 
                 desc="Training Epochs", 
                 ncols=100):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                # batch = (name, time_x, batch_series_x, batch_series_noisy_x, time_y, batch_series_y, batch_series_noisy_y)
                patient_id = batch["name"]
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float().to(self.device, non_blocking=True)
                #print('--------Shape of model input:--------',batch_x.shape)
                #print('--------Shape of model target:--------',batch_y.shape)
                if self.args.use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        if 'former' in self.args.model:
                            outputs = self.model(batch_x)
                            outputs = outputs.unsqueeze(-1)  # [batch, pred_len, 1]  
                        else:
                            outputs = self.model(batch_x)

                        if self.args.MO_flag:
                            outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                        else:
                            outputs = outputs[:, -1:, -self.args.output_dim:]
                            batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)                      
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'former' in self.args.model:
                        outputs = self.model(batch_x)
                        outputs = outputs.unsqueeze(-1)  # [batch, pred_len, 1]  
                    else:
                        outputs = self.model(batch_x)

                    if self.args.MO_flag:
                        outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                    else:
                        outputs = outputs[:, -1:, -self.args.output_dim:]
                        batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)                      

                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
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
                    time_now = time.time()

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
                

            tqdm.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            if best_vali_loss is None:
                best_vali_loss = vali_loss
                training_end_time = time.time()

            elif best_vali_loss > vali_loss:
                best_vali_loss = vali_loss
                training_end_time = time.time()

            cost_time_epoch = time.time() - time_now_epoch
            wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": vali_loss,
            "cost_time": cost_time_epoch}, commit=True)

            tqdm.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            if not self.args.is_hyperopt:
                if (epoch + 1) % 50 == 0:
                    checkpoint_path = os.path.join(path, f'checkpoint_epoch{epoch + 1}_tran_loss{train_loss}_val_loss{vali_loss}.pth')
                    torch.save(self.model, checkpoint_path)
                    tqdm.write(f"Saved checkpoint at epoch {epoch + 1}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                #tqdm.write("Early stopping")
                #logger.info(f"Early stopping")
                logger.info(f"Early stopped at epoch {epoch} Best vali loss {best_vali_loss:.5f}")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                tqdm.write('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
        training_time = training_end_time - time_start
        print(f"Training cost time: {training_time:.2f}s")
        logger.info(f"Training cost time: {training_time:.2f}s")

        if self.args.is_save_checkpoint:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model = torch.load(best_model_path)
            logger.info(f"Loaded best checkpoint")
        
            df = pd.DataFrame(train_log)
            csv_path = os.path.join(path, 'train_log.csv') 
            df.to_csv(csv_path, index=False)
        wandb.log({"total_cost_time": training_time})
        wandb.log({"best_vali_loss": best_vali_loss})
        logger.info(f"-------------------End training-------------------------")
        return self.model, best_vali_loss , training_time
    
    def vali(self, vali_loader, criterion):
        vali_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float()

                time_now = time.time() 
            
                if 'former' in self.args.model:
                    outputs = self.model(batch_x)
                    outputs = outputs.unsqueeze(-1)  # [batch, out_seq_len, 1]  
                else:
                    outputs = self.model(batch_x)
                if self.args.MO_flag:
                    outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                else:
                    outputs = outputs[:, -1:, -self.args.output_dim:]
                    batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)                      

                speed = (time.time() - time_now)
                #tqdm.write('\tspeed: {:.8f}s/iter'.format(speed))
                
                # outputs = outputs[:, -self.args.pred_len:, -1:]
                # batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device, non_blocking=True)

                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()

                # loss = criterion(pred, true)

                loss = criterion(outputs, batch_y)
                # vali_loss.append(loss)
                vali_loss.append(loss.item()) 
        vali_loss = np.average(vali_loss)
        self.model.train()
        return vali_loss
    
    def test(self, setting, model = None, load_model_path = None):
        logger.info(f"-------------------Start testing-------------------------")
        test_loader = self._get_test_data()
        if model is not None:
            self.model = model
        else:
            try:
                if load_model_path:
                    print(f'Loading model from: {load_model_path}')
                    self.model = torch.load(load_model_path, map_location=self.device)
                    logger.info(f"Loading model from: {load_model_path}")
                else:
                    default_path = os.path.join(self.args.checkpoints, self.args.project, setting, 'checkpoint.pth')
                    print(f'Loading model from default path: {default_path}')

                    #self.model = torch.load(default_path, map_location=self.device)

                    #self._load_new_hyper_para(setting) 

                    self.model = torch.load(default_path, map_location=self.device)
                    self.model.to(self.device)

                    logger.info(f"Loading model from optimzed: {default_path}")
                print("Model loaded successfully")
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found: {e.filename}") from e
            except Exception as e:
                raise RuntimeError(f"Error loading model: {str(e)}") from e

        preds_by_patient = defaultdict(list)
        trues_by_patient = defaultdict(list)
        no_preds_by_patient = defaultdict(list)
        times_by_patient = defaultdict(list)
        patient_ids = set()

        self.model.eval()
        timings = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader),total=len(test_loader), 
                                desc="Testing progress", 
                                ncols=100,):
                # if i> 500:
                #     break
                batch_x = batch["series_x"].float().to(self.device, non_blocking=True)
                batch_y = batch["series_y"].float().to(self.device, non_blocking=True)
                time_stamp = batch["time_y"].float().to(self.device, non_blocking=True)

                patient_id = batch["name"][0]
                start_time = time.perf_counter()
            
                if 'former' in self.args.model:
                    outputs = self.model(batch_x)
                    outputs = outputs.unsqueeze(-1)
                else:
                    outputs = self.model(batch_x)
                if self.args.MO_flag:
                    outputs = outputs[:, -self.args.pred_len:, -self.args.output_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, -self.args.output_dim:].to(self.device, non_blocking=True)
                    no_pred = batch_x[:, -self.args.pred_len:, -self.args.output_dim:]
                else:
                    outputs = outputs[:, -1:, -self.args.output_dim:]
                    batch_y = batch_y[:, -1:, -self.args.output_dim:].to(self.device, non_blocking=True)                      
                    no_pred = batch_x[:, -1:, -self.args.output_dim:]
                torch.cuda.synchronize()  
                end_time = time.perf_counter()
                timings.append((end_time - start_time) * 1000)

                outputs = outputs[:, -1:, -self.args.output_dim::]
                batch_y = batch_y[:, -1:, -self.args.output_dim::] # .to(self.device, non_blocking=True)
                # outputs = outputs[:, -self.args.pred_len:, -1:]
                # batch_y = batch_y[:, -self.args.pred_len:, -1:] # .to(self.device, non_blocking=True)
                no_pred = batch_x[:, -1:, -self.args.output_dim::]
                time_stamp = time_stamp[:, -1:, -self.args.output_dim::]
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                no_pred = no_pred.detach().cpu().numpy()
                time_stamp = time_stamp.detach().cpu().numpy()
                preds_by_patient[patient_id].append(pred)
                trues_by_patient[patient_id].append(true)
                no_preds_by_patient[patient_id].append(no_pred)
                times_by_patient[patient_id].append(time_stamp)
                patient_ids.add(patient_id)
        
        avg_inference_time = sum(timings) / len(timings)
        std_inference_time = torch.std(torch.tensor(timings)).item()
        if self.args.test_flop:
            model_params, macs, params = test_params_flop(self.model,(batch_x.shape[1],batch_x.shape[2]))
            logger.info('INFO: Trainable parameter count: {:.5f}M'.format(model_params / 1000000.0))
            logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
            #exit()
        logger.info(f'Average inference time {avg_inference_time}')
        logger.info(f'Standard devation inference time {std_inference_time}')

        # result save
        test_result_folder_path = os.path.join(self.args.checkpoints, self.args.project, setting, 'test_results')
        os.makedirs(test_result_folder_path, exist_ok=True)
        patient_ids = list(patient_ids) 
        result_path = os.path.join(test_result_folder_path, "result.txt")
        with open(result_path, 'a') as f:
            f.write(f"{setting}\n\n")
            rmse_list = []
            re_rmse_list = []
            patient_re_rmse_list = []
            for patient in patient_ids:
                
                preds = np.stack(preds_by_patient[patient])
                trues = np.stack(trues_by_patient[patient])
                no_preds = np.stack(no_preds_by_patient[patient])
                times_pred = np.stack(times_by_patient[patient])
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                no_preds = no_preds.reshape(-1, no_preds.shape[-2], no_preds.shape[-1])
                times_pred = times_pred.reshape(-1, times_pred.shape[-2], times_pred.shape[-1])

                np.save(os.path.join(test_result_folder_path, f"{patient}_pred.npy"), preds)
                np.save(os.path.join(test_result_folder_path, f"{patient}_true.npy"), trues)
                np.save(os.path.join(test_result_folder_path, f"{patient}_x.npy"), no_preds)
                np.save(os.path.join(test_result_folder_path, f"{patient}_time.npy"), times_pred)
                mae, mse, rmse, mape, mspe, rse, corr, relative_rmse = metric(preds, trues, no_preds)
                rmse_list.append(rmse)
                re_rmse_list.append(relative_rmse)
                patient_re_rmse_list.append((patient,relative_rmse))

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
            rmse_mean = np.mean(rmse_list)
            rmse_std = np.std(rmse_list)
            rmse_median = np.median(rmse_list)
            rmse_q1, rmse_q3 = np.percentile(rmse_list, [25, 75])

            re_rmse_mean = np.mean(re_rmse_list)
            re_rmse_std = np.std(re_rmse_list)
            re_rmse_median = np.median(re_rmse_list)
            re_rmse_q1, re_rmse_q3 = np.percentile(re_rmse_list, [25, 75])

            print("\n=== Inference time Statistics ===")
            print(f'Average inference time {avg_inference_time}')
            print(f'Standard devation inference time {std_inference_time}')

            print("\n=== Patient-wise RMSE Statistics ===")
            print(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}")
            print(f"Median: {rmse_median:.7f}")
            print(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]")

            print("\n=== Patient-wise Relative RMSE Statistics ===")
            print(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}")
            print(f"Median: {re_rmse_median:.7f}")
            print(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]")
            
            f.write("\n=== Inference time Statistics ===\n")
            f.write(f"Average: {avg_inference_time:.7f} ms\n")  
            f.write(f"Std Dev: {std_inference_time:.7f} ms\n")
            f.write(f'Trainable parameter count: {model_params/1e6:.2f}M\n')
            f.write(f"{'Computational complexity:':<30} {macs}\n")
            f.write(f"{'Number of parameters:':<30} {params}\n")
            logger.info(f'Average Inference time {avg_inference_time}')

            f.write("\n=== Patient-wise RMSE Statistics ===\n")
            f.write(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n")
            f.write(f"Median: {rmse_median:.7f}\n")
            f.write(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]\n\n")
            logger.info(f'Patient-wise RMSEe Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n')

            f.write("=== Patient-wise Relative RMSE Statistics ===\n")
            f.write(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n")
            f.write(f"Median: {re_rmse_median:.7f}\n")
            f.write(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]\n\n")
            logger.info(f'Patient-wise Relative RMSE Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n')

            rmse_path = os.path.join(test_result_folder_path, "rmse_list.npy")
            re_rmse_path = os.path.join(test_result_folder_path, "re_rmse_list.npy")
            np.save(rmse_path, np.array(rmse_list))
            np.save(re_rmse_path, np.array(re_rmse_list))

            sorted_by_re_rmse = sorted(patient_re_rmse_list, key=lambda x: x[1])  
            top5_best = sorted_by_re_rmse[:5]
            top5_worst = sorted_by_re_rmse[-5:][::-1]
            print("\nTop 5 best patients (lowest relative RMSE):")
            f.write("=== Top 5 Patients (Lowest Relative RMSE) ===\n")
            logger.info("=== Top 5 Patients (Lowest Relative RMSE) ===")
            for i, (pid, rermse) in enumerate(top5_best, 1):
                print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
                f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
                logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")

            print("\nWorst 5 patients (highest relative RMSE):")
            f.write("\n=== Worst 5 Patients (Highest Relative RMSE) ===\n")
            logger.info("=== Worst 5 Patients (Highest Relative RMSE) ===")
            for i, (pid, rermse) in enumerate(top5_worst, 1):
                print(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.4f}")
                f.write(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}\n")
                logger.info(f"# {i}, Patient ID: {pid}, Relative RMSE: {rermse:.7f}")
            wandb.log({
            "rmse_mean": rmse_mean,
            "re_rmse_mean": re_rmse_mean,
            "inference_time": avg_inference_time}, commit=True)
            logger.info(f"-------------------End testing-------------------------")
        return rmse_mean, re_rmse_mean, avg_inference_time, model_params, macs, params

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
    