from exp.exp_main import Exp_Main
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from models.Many2Many_benchmark import Model
from models.DLinear_benchmark import DecompLinear

from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop
from utils.metrics import metric
from resp_db.data_loader_benchmark import Create_dataset_population
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
        _,_,test_loader = Create_dataset_population(self.args)
        return test_loader
    def convert_to_model_input(self,batch_x, seq_len=12, input_features=4):
        """
        将 (batch, 32, 1) 的 batch_x 转换为 (batch, seq_len, input_features) 的模型输入。
        例如：从 32 个时间步中截取最后 4 个时间步，并为每个时间步填充 4 个特征。
        """
        batch, total_len, _ = batch_x.shape
        # 取最后 seq_len 个时间步（避免补零）
        start_idx = total_len - seq_len

        # 为每个时间步构造 input_features 维特征
        expanded = torch.zeros(batch, seq_len, input_features).to(batch_x.device)
        for b in range(batch):
            for i in range(seq_len):
                # 取当前及之前的 (input_features-1) 个点
                start_feature = max(0, start_idx + i - (input_features - 1))
                features = batch_x[b, start_feature : start_idx+i+1, 0]  # shape=(n,)
                # 不足时左侧补零
                if len(features) < input_features:
                    features = torch.cat([
                        torch.zeros(input_features - len(features)).to(batch_x.device),
                        features
                    ])
                expanded[b, i, :] = features
        return expanded
    # def predict_future(self, model, initial_input, pred_len=24):
    #     current_input = initial_input  # shape=(1, 4, 4)
    #     print('==================',current_input[:, 1:, :].shape)
    #     predictions = []
    #     for _ in range(pred_len):
    #         next_pred = model(current_input)[:, -1:, :]  # shape=(1, 1, 1)
    #         predictions.append(next_pred)
    #         print('===123=========',next_pred.shape)
    #         # 更新输入：去掉最旧时间步，加入最新预测
    #         new_features = torch.cat([
    #             current_input[:, 1:, :],                     # shape=(1, 3, 4)
    #             next_pred.expand(-1, -1, 4).unsqueeze(1)     # shape=(1, 1, 4)
    #         ], dim=1)  # -> shape=(1, 4, 4)

    #         current_input = new_features
    #     return torch.cat(predictions, dim=1)  # shape=(1, 24, 1)


    def test(self, setting, load_model_path=None):
        test_loader = self._get_data()
        if load_model_path:
            print(f'Loading existing model from: {load_model_path}')
            #LSTM
            #checkpoint = torch.load(load_model_path, map_location=self.device)
            model_dict = {
            'DLinear': DLinear,
            'MLP': MLP, # Pure 3 layer LSTM
            'XGBoostTSF': XGBoostTSF,
            'wLMS': wLMS,
            'WaveletLSTM': WaveletLSTM,
            'LGEANet': LGEANet,
            'Lin': Lin, 
            'TransformerTSFv2':TransformerTSFv2,
            'Many2Many':Many2Many,
                }
            self.model = model_dict[self.args.model].Model(self.args).float()
           

            self.model.load_state_dict(torch.load(load_model_path, weights_only=True))
            self.model.to(self.device, map_location=self.device)
            # DLinear
            # checkpoint = torch.load(load_model_path, map_location=self.device)
            # print(checkpoint)
            # required_args = {
            #     'seq_len': checkpoint['input_features'],
            #     'pred_len': checkpoint['future_steps'],
            #     'individual': True,
            #     'enc_in': 1
            # }
            # self.model = DecompLinear(**required_args)
            # state_dict = checkpoint['model_state_dict']
            # self.model.load_state_dict(state_dict)
            # self.model.to(self.device)

        else:
            model_state_dict_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            #print(f'Loading model from default path: {model_state_dict_path}')
            state_dict = torch.load(model_state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

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
                # DLINEAR
                #print('asdasdasdasdasd',batch_x.shape)
                batch_x = batch_x.permute(0, 2, 1)
                #print('Input shape:',batch_x.shape)
                #batch_x = self.convert_to_model_input(batch_x, seq_len=12, input_features=32)
                #print('Input of model',batch_x.shape)  # 输出: torch.Size([1, 4, 4])
                
                patient_id = batch["name"][0]
                if self.args.use_amp:
                    with torch.amp.autocast():
                        if 'former' in self.args.model:
                            print('Remained to add Transformer')
                        else:
                            outputs = self.model(batch_x)
                else:
                    if 'former' in self.args.model:
                        if self.args.output_attention:
                            print('Remained to add Transformer')
                    else:
                        outputs = self.model(batch_x)
                        #print('Output of model',outputs.shape)  # 输出: torch.Size([1, 4, 4])

                outputs = outputs[:, -1:, -1:]
                batch_y = batch_y[:, -1:, -1:] # .to(self.device)
                # outputs = outputs[:, -self.args.pred_len:, -1:]
                # batch_y = batch_y[:, -self.args.pred_len:, -1:] # .to(self.device)
                no_pred = batch_x[:, -1:, -1:]
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                no_pred = no_pred.detach().cpu().numpy()
                #print('RMSE:',np.sqrt((pred - true) ** 2))
                # print('1231231231',patient_id)
                # print('123123213',pred)
                preds_by_patient[patient_id].append(pred)
                trues_by_patient[patient_id].append(true)
                no_preds_by_patient[patient_id].append(no_pred)
                patient_ids.add(patient_id)

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
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
                index = np.arange(len(preds)) / 26  
                prediction_horizon_ms = int(self.args.pred_len * 1000 / 26)

                plt.figure(figsize=(12, 6))
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
            re_rmse_mean = np.mean(re_rmse_list)     
            re_rmse_std = np.std(re_rmse_list)
            print(f"\nPatient-wise RMSE mean: {rmse_mean:.7f}")
            print(f"\nPatient-wise RMSE std: {rmse_std:.7f}")
            f.write(f"\nPatient-wise RMSE mean: {rmse_mean:.7f}\n")
            f.write(f"Patient-wise RMSE std: {rmse_std:.7f}\n\n")

            print(f"\nPatient-wise relative RMSE mean: {re_rmse_mean:.7f}")
            print(f"\nPatient-wise relative RMSE std: {re_rmse_std:.7f}")
            f.write(f"\nPatient-wise relative RMSE mean: {re_rmse_mean:.7f}\n")
            f.write(f"Patient-wise relative RMSE std: {re_rmse_std:.7f}\n\n")


        return
    
    
