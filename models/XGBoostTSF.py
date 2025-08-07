__all__ = ['XGBoostTSF']

import torch
import argparse
import xgboost as xgb
import argparse
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import json

np.set_printoptions(suppress=True) 

class Model(nn.Module):
    """XGBOOST for time series forecasting.

    Every element of the sliding input window is considered as a
    separate feature. This concept was proposed by Elsayed et al 2021
    (https://doi.org/10.48550/arXiv.2101.02118).
    """

    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        self.n_estimators = args.n_estimators
        self.max_depth = args.max_depth
        self.subsample_baselearner = args.subsample_baselearner
        self.gamma = args.gamma
        self.DMS_flag = args.DMS_flag
        self.min_child_weight = args.min_child_weight
        self.learning_rate = args.learning_rate
        self.reg_lambda = args.reg_lambda
        self.model = xgb.XGBRegressor(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample_baselearner,
            scale_pos_weight=1,
            verbosity=2,
            n_jobs=-1,
            tree_method="hist",
            device='cuda:0',
            seed=args.random_seed,
            dtype=np.float32,
        )
        self.trained = False

    @staticmethod
    def plot_feature_importance(model):
        fig, axis = plt.subplots(1, 1)
        axis.bar(range(len(model.feature_importances_)), model.feature_importances_)
        return fig
    
    def save(self,model_path):
        self.model.save_model(model_path)

    def load(self,model_path):
        if model_path.endswith('.json'):
            booster = xgb.Booster()
            booster.load_model(model_path)
            self.model._Booster = booster
        else:
            self.model.load_model(model_path)
        self.trained = True
        
    def forward(self, features, targets):
        self.model.fit(features, targets)
        self.trained = True

        if self.DMS_flag: # For DMS 
            output = self.model.predict(features)
        else:
            # Recursive multi-step prediction
            pred_len = self.args.pred_len
            lookback = features.shape[1]
            output = []

            for i in range(features.shape[0]):  # loop over batch
                seq = list(features[i])  # initial input sequence
                preds = []
                for _ in range(pred_len):
                    x_input = np.array(seq[-lookback:]).reshape(1, -1)
                    next_val = self.model.predict(x_input)[0]
                    preds.append(next_val)
                    seq.append(next_val)
                output.append(preds)
        return output

    def predict_(self, features: torch.Tensor) -> torch.Tensor:
        if not self.trained:
            raise ValueError("Model not trained!")
        outputs = self.model.predict(features)
        return outputs

if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser(description='XGBoost for Time Series Forecasting')
        parser.add_argument('--n_estimators', type=int, default=100)
        parser.add_argument('--max_depth', type=int, default=6)
        parser.add_argument('--subsample_baselearner', type=float, default=0.8)
        parser.add_argument('--gamma', type=float, default=0.1)
        parser.add_argument('--min_child_weight', type=float, default=1)
        parser.add_argument('--reg_lambda', type=float, default=1)

        parser.add_argument('--learning_rate', type=float, default=0.1)
        parser.add_argument('--DMS_flag', default= True, help='Flag True for DMS or False for IMS')
        parser.add_argument('--pred_len', type=int, default=24, help="预测序列长度,可为0")

        return parser.parse_args()

    args = get_args()
    model = Model(args)

    # 构造模拟数据（batch_size=16，sequence_len=10，seq_len=1）
    batch_size = 16
    sequence_len = 10
    seq_len = 1
    output_dim = 1  # 预测单值

    X = torch.randn(batch_size, sequence_len * seq_len)
    y = torch.randn(batch_size, output_dim)

    # 转为 numpy（XGBoost 不支持 tensor）
    #output = model(X.numpy(), y.numpy())
    output = model(X, y)
    print("预测输出维度：", output.shape)
    print("预测值：", output)
