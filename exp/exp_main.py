from exp.exp_basic import Exp_Basic
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many
from utils.tools import EarlyStopping, adjust_learning_rate, visual,visual_all, test_params_flop
from utils.metrics import metric
from resp_db.data_loader import Create_dataset_population
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.args = args

    def _build_model(self):
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
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        train_loader, val_loader = Create_dataset_population(self.args)
        return train_loader, val_loader
        
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    

    def train(self, setting):
        train_loader , vali_loader  = self._get_data(self.args)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (name, time_x, batch_series_x, batch_series_noisy_x, time_y, batch_series_y, batch_series_noisy_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_series_x.float().to(self.device)
                batch_y = batch_series_y.float().to(self.device)
              
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'former' in self.args.model:
                            print('Remained to add Transformer')
     
                        else:
                            outputs = self.model(batch_x)

                        outputs = outputs[:, -self.args.pred_len:, -1:]
                        batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'former' in self.args.model:
                        if self.args.output_attention:
                            print('Remained to add Transformer')
                    else:
                        outputs = self.model(batch_x)
                    
                    print("123 shape of outputs:",outputs.shape)
                    print("123 shape of batch_y:",batch_y.shape)                    
                    outputs = outputs[:, -self.args.pred_len:, -1:]
                    batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)
                    print("456 shape of outputs:",outputs.shape)
                    print("456 shape of batch_y:",batch_y.shape)    
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 1000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.8f}s/iter; left time: {:.4f}s'.format(speed, left_time))
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (name, time_x, batch_series_x, batch_series_noisy_x, time_y, batch_series_y, batch_series_noisy_y) in enumerate(vali_loader):
                batch_x = batch_series_x.float().to(self.device)
                batch_y = batch_series_y.float()

                time_now = time.time() 
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'former' in self.args.model:
                            print('Remained to add Transformer')
                        else:
                            outputs = self.model(batch_x)
                else:
                    if 'former' in self.args.model:
                        print('Remained to add Transformer')

                    else:
                        outputs = self.model(batch_x)
                speed = (time.time() - time_now)
                print('\tspeed: {:.8f}s/iter'.format(speed))
                
                outputs = outputs[:, -self.args.pred_len:, -1:]
                batch_y = batch_y[:, -self.args.pred_len:, -1:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    