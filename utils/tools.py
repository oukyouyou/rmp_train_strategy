import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import logging
from utils.logger import LoggerMixin
import dill
import os
import json

logger = logging.getLogger(__name__)
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=False):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 2 else args.learning_rate * (0.9 ** ((epoch - 2) // 1))}
    elif args.lradj == 'type4':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * (0.99 ** ((epoch - 10) // 5))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping(LoggerMixin):
    def __init__(self, patience=7, verbose=False, delta=0, is_save_checkpoint = False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.is_save_checkpoint = is_save_checkpoint

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.is_save_checkpoint:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                #torch.save(model, path + "/best_model.pth")
        else:
            self.best_score = score
            if self.is_save_checkpoint:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        with open(path + '/' + 'checkpoint.pth', 'wb') as f:
            torch.save(model, f)
            f.flush()                  
            os.fsync(f.fileno()) 
        #torch.save(model, path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_all(true, preds=None,input_unprediction=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    shape = np.shape(true)
    true = np.reshape(true,[shape[0]*shape[2],3])
    preds = np.reshape(preds,[shape[0]*shape[2],3])
    input_unprediction=  np.reshape(input_unprediction,[shape[0]*shape[2],3])
    plt.figure()
    plt.plot(true[100:400,0], label='GroundTruth_x', linewidth=2)
    plt.plot(input_unprediction[100:400,0], label='Unprediction', linewidth=2)
    if preds is not None:
        plt.plot(preds[100:400,0], label='Prediction', linewidth=2,ls = '--')
    plt.legend()
    plt.savefig(name)

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        #print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        #print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        #print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return model_params, macs, params 

def read_hyperopt_object(filepath: str, settings: str, json_filename="trial_results.json"):
    hyperopt_filepath = os.path.join(filepath, settings+ ".hyperopt")
    with open(hyperopt_filepath, "rb") as f:
        trials = dill.load(f)
    best_trial = trials.best_trial
    summary = (
        "Found saved Trials! \n "
        "------------- Information --------------------\n"
        f"Setting name: {best_trial['result']['settings']} \n"
        f"Number of evaluated combinations: {len(trials.trials)} \n"
        f"Best MSE loss on val set: {best_trial['result']['loss']} \n "
        f"Corresponding hyperparas: {best_trial['result']['applied_hyper_paras']} \n"
        "------------- End Information ----------------"
    )
    save_para = {
        "settings": best_trial['result']['settings'],
        "vali loss": best_trial['result']['loss'] ,
        "hyper_para": best_trial['result']['applied_hyper_paras']
    }
    print(summary)
    logger.info(summary)
    json_path = os.path.join(filepath, json_filename)
    with open(json_path, "w") as f:
        json.dump(save_para, f, indent=4) 
    return best_trial