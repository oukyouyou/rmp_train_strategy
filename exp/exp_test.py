from exp.exp_main import Exp_Main
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop
from utils.metrics import metric
from resp_db.data_loader import Create_dataset_individual, Create_dataset_population
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from collections import defaultdict
from tqdm import tqdm
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Test(Exp_Main):
    def __init__(self, args):
        super(Exp_Test, self).__init__(args)
        self.args = args

    def _get_data(self):
        test_loader = Create_dataset_individual(self.args)
        return test_loader

    def test(self, setting, model = None, load_model_path = None):
        test_loader = self._get_data()
        if model is not None:
            self.model = model
        else:
            try:
                if load_model_path:
                    print(f'Loading model from: {load_model_path}')
                    self.model = torch.load(load_model_path, map_location=self.device)
                else:
                    default_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
                    print(f'Loading model from default path: {default_path}')
                    state_dict = torch.load(default_path, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                print("Model loaded successfully")
                
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Model file not found: {e.filename}") from e
            except Exception as e:
                raise RuntimeError(f"Error loading model: {str(e)}") from e

        preds_by_patient = defaultdict(list)
        trues_by_patient = defaultdict(list)
        no_preds_by_patient = defaultdict(list)
        patient_ids = set()

        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_loader),total=len(test_loader), 
                                desc="Testing progress", 
                                ncols=100,):
                batch_x = batch["series_noisy_x"].float().to(self.device)
                batch_y = batch["series_noisy_y"].float().to(self.device)
                patient_id = batch["name"][0]
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

                outputs = outputs[:, -1:, -1:]
                batch_y = batch_y[:, -1:, -1:] # .to(self.device)
                # outputs = outputs[:, -self.args.pred_len:, -1:]
                # batch_y = batch_y[:, -self.args.pred_len:, -1:] # .to(self.device)
                no_pred = batch_x[:, -1:, -1:]
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                no_pred = no_pred.detach().cpu().numpy()
                # print('1231231231',patient_id)
                # print('123123213',pred)
                preds_by_patient[patient_id].append(pred)
                trues_by_patient[patient_id].append(true)
                no_preds_by_patient[patient_id].append(no_pred)
                patient_ids.add(patient_id)

        if self.args.test_flop:
            test_params_flop(self.model,(batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        # result save
        folder_path = self.args.results_path + setting + '/' 
        os.makedirs(folder_path, exist_ok=True)
        patient_ids = list(patient_ids) 
        result_path = os.path.join(folder_path, "result.txt")
        with open(result_path, 'a') as f:
            f.write(f"{setting}\n\n")

            rmse_list = []
            re_rmse_list = []
            for patient in patient_ids:
                preds = np.stack(preds_by_patient[patient])
                trues = np.stack(trues_by_patient[patient])
                no_preds = np.stack(no_preds_by_patient[patient])

                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                no_preds = no_preds.reshape(-1, no_preds.shape[-2], no_preds.shape[-1])

                np.save(os.path.join(folder_path, f"{patient}_pred.npy"), preds)
                np.save(os.path.join(folder_path, f"{patient}_true.npy"), trues)
                np.save(os.path.join(folder_path, f"{patient}_x.npy"), no_preds)

                mae, mse, rmse, mape, mspe, rse, corr, relative_rmse = metric(preds, trues, no_preds)
                rmse_list.append(rmse)
                re_rmse_list.append(relative_rmse)
                print(f"Patient: {patient}, mse: {mse:.7f}, rmse: {rmse:.7f}, relative_rmse: {relative_rmse:.7f}")
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
                plt.savefig(os.path.join(folder_path, f"{patient}_plot.png"))
                plt.close()
            rmse_mean = np.mean(rmse_list)
            rmse_std = np.std(rmse_list)
            rmse_median = np.median(rmse_list)
            rmse_q1, rmse_q3 = np.percentile(rmse_list, [25, 75])

            re_rmse_mean = np.mean(re_rmse_list)
            re_rmse_std = np.std(re_rmse_list)
            re_rmse_median = np.median(re_rmse_list)
            re_rmse_q1, re_rmse_q3 = np.percentile(re_rmse_list, [25, 75])

            print("\n=== Patient-wise RMSE Statistics ===")
            print(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}")
            print(f"Median: {rmse_median:.7f}")
            print(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]")

            print("\n=== Patient-wise Relative RMSE Statistics ===")
            print(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}")
            print(f"Median: {re_rmse_median:.7f}")
            print(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]")

            f.write("\n=== Patient-wise RMSE Statistics ===\n")
            f.write(f"Mean ± Std: {rmse_mean:.7f} ± {rmse_std:.7f}\n")
            f.write(f"Median: {rmse_median:.7f}\n")
            f.write(f"IQR: [{rmse_q1:.7f}, {rmse_q3:.7f}]\n\n")

            f.write("=== Patient-wise Relative RMSE Statistics ===\n")
            f.write(f"Mean ± Std: {re_rmse_mean:.7f} ± {re_rmse_std:.7f}\n")
            f.write(f"Median: {re_rmse_median:.7f}\n")
            f.write(f"IQR: [{re_rmse_q1:.7f}, {re_rmse_q3:.7f}]\n\n")

            rmse_path = os.path.join(folder_path, "rmse_list.npy")
            re_rmse_path = os.path.join(folder_path, "re_rmse_list.npy")
            np.save(rmse_path, np.array(rmse_list))
            np.save(re_rmse_path, np.array(re_rmse_list))


        return
    
