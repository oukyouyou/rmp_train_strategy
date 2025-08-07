import argparse
from pathlib import Path


from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many, Mamba, TimesNet
from models.Stat_models import Arima

MODEL_DICT = {
            'DLinear': DLinear,
            'MLP': MLP, 
            'XGBoostTSF': XGBoostTSF,
            'wLMS': wLMS,
            'WaveletLSTM': WaveletLSTM,
            'LGEANet': LGEANet,
            'LSTM': Lin, 
            'TransformerTSFv2':TransformerTSFv2,
            'Many2Many':Many2Many,
            'Arima': lambda args: Arima(args),
            'wLMS':wLMS,
            'Mamba':Mamba,
            'TimesNet':TimesNet,
        }

class Benchmark_args():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='New benchmark for RMP')
        self._setup_arguments()
        self.args = None
    def _setup_arguments(self):
        self.parser = argparse.ArgumentParser(description='New benchmark for RMP')
        # random seed
        self.parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
        # basic config
        self.parser.add_argument('--is_training', type=int,  default=1, help='status')
        self.parser.add_argument('--is_fine_tuning', type=bool,  default=False, help='status')
        self.parser.add_argument('--is_hyperopt', type=bool,  default=True, help='status')
        self.parser.add_argument('--is_save_checkpoint', type=bool, default=True)
        self.parser.add_argument('--db-root', type=Path, default="/home/wang-jiaoyang/open_access_rpm_signals_master.db")
        self.parser.add_argument('--table_name')

        self.parser.add_argument('--project', choices=['PL','PS-4DCT', 'PS', 'test'], default='PL')
        self.parser.add_argument('--patient_id', type=str, default='418', help="格式如 '010', 3 digits")
        self.parser.add_argument('--f_num', type=int, default='1', help="格式 1 digit, 0 to 8")

        #self.parser.add_argument('--Fx_flag', type=str, default='Fx5', help="仅patient-specific项目使用,格式如 Fx2")
        self.parser.add_argument('--model', type=str, default='LSTM',
                            help='model name, options: [LSTM, Many2Many, MLP, TransformerTSFv2, DLinear') # , TiDE, Lin, MP, LSTfromer

        self.parser.add_argument('--DMS_flag', default= True, help='Flag true for DMS or false for IMS')
        self.parser.add_argument('--MO_flag', default= False, help='Flag for multi-output or single-output')
        self.parser.add_argument('--pretrained_path', default= '', help='Pre-trainde model path')
        
        # input size
        self.parser.add_argument('--input_size', type=int, default=1, help='input features of LSTM (m)')
        self.parser.add_argument('--seq_len', type=int, default=32, help="历史序列长度")
        self.parser.add_argument('--label_len', type=int, default=16, help="标签序列长度,可为0")
        self.parser.add_argument('--pred_len', type=int, default=24, help="预测序列长度,可为0")
        self.parser.add_argument('--only-beam-on', action='store_true', default=True, help="仅保留beam-on时段数据")
        self.parser.add_argument('--remove-offset', action='store_true', default=True, help="移除信号偏移量")

        # LSTM
        self.parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of LSTM')
        self.parser.add_argument('--num_layers', type=int, default=8, help='Number of LSTM layers')
        
        # DLinear
        self.parser.add_argument('--kernel_size', type=int, default=25, help='Decompsition Kernel Size for linear')
        self.parser.add_argument('--individual', action='store_true', default=True, help='individual for seasonal and trend decomp')

        # MLP
        self.parser.add_argument('--mlp_num_layers', type=int, default=3, help='Number of layers in MLP')  # 1层、2层、3层等
        self.parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MLP')
        # activation function
        self.parser.add_argument('--use_revin', type=bool, default=False, help='Enable RevIN normalization')
        self.parser.add_argument('--activation_function', type=str, default='relu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'Dyt'],
                            help='Activation function to use in the model')
        
        # TransformerTSFv2
        self.parser.add_argument('--layer_dim_val', type=int, default=32)
        self.parser.add_argument('--n_decoder_layers', type=int, default=2)
        self.parser.add_argument('--n_encoder_layers', type=int, default=2)
        self.parser.add_argument('--n_heads', type=int, default=4)

        # XGBoost
        self.parser.add_argument('--n_estimators', type=int, default=100)
        self.parser.add_argument('--max_depth', type=int, default=6)
        self.parser.add_argument('--subsample_baselearner', type=float, default=0.8)
        self.parser.add_argument('--gamma', type=float, default=0.1)
        self.parser.add_argument('--min_child_weight', type=float, default=1)
        self.parser.add_argument('--reg_lambda', type=float, default=1)

        # wLMS
        self.parser.add_argument('--wavelet', type=str, default='sym2',help='Symlet-2 wavelet')
        self.parser.add_argument('--wLMS_J', type=int, default=7, help='Decomposition levels')
        self.parser.add_argument('--wLMS_a', type=float, default=None)
        self.parser.add_argument('--wLMS_M', type=int, default=2)
        self.parser.add_argument('--wLMS_mu', type=float, default=1)
        
        # Mamba
        self.parser.add_argument('--d_model', type=int, default=8, help='Model dimension d_model')
        self.parser.add_argument('--d_state', type=int, default=16, help='SSM state expansion factor')
        self.parser.add_argument('--d_conv', type=int, default=4, help='Local convolution width')
        self.parser.add_argument('--expand', type=int, default=2, help='Block expansion factor')

        # TimesNet
        self.parser.add_argument('--top_k', type=int, default=3, help='top_k of fft')
        self.parser.add_argument('--d_ff', type=int, default=3, help='Feed-Forward Dimension')
        self.parser.add_argument('--num_kernels', type=int, default=3, help='cnn kernel filters')
        self.parser.add_argument('--norm_flag', default=True, help='flage of normalization at begining')

        # General
        self.parser.add_argument('--batch_size', type=int, default=256)
        self.parser.add_argument('--data_shuffle', type=bool, default=False)
        self.parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
        self.parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
        self.parser.add_argument('--val_ratio', type=float, default=0.7, help='train ratio = val_ratio/ val ratio = 1 - val_ratio')
        self.parser.add_argument('--split_ratio', type=float, default=0.7, help='ratio to split dataset for test')
              
        self.parser.add_argument('--data_specific_train', type=int, default=20, help='initial x s for data-specific')
        self.parser.add_argument('--prefetch_factor', default=8, help='data loader prefetch_factor')
        self.parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
        self.parser.add_argument('--fourier_smoothing_hz', type=int, default=1)
        self.parser.add_argument('--white_noise_db', type=int, default=27)
        self.parser.add_argument('--sampling_rate_hz', type=int, default=26)
        self.parser.add_argument('--apply_denoise', action='store_true', default=False)
        self.parser.add_argument('--apply_noise', action='store_true', default=False)
        self.parser.add_argument('--apply_scaling', action='store_true', default=True)
        self.parser.add_argument('--scaling_period_s', type=tuple[float, float] , default=(0, 20))

        self.parser.add_argument('--checkpoints', type=str, default='/mnt/nas-wang/nas-ssd/Results/RMP/', help='location of model checkpoints')
        
        # optimization
        self.parser.add_argument('--itr', type=int, default=1, help='experiments times')
        self.parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
        self.parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
        self.parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
        self.parser.add_argument('--des', type=str, default='test', help='exp description')
        self.parser.add_argument('--loss', type=str, default='mse', help='loss function')
        self.parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
        self.parser.add_argument('--pct_start', type=float, default=0.5, help='pct_start')
        self.parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

        # GPU
        self.parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        self.parser.add_argument('--gpu', type=int, default=0, help='gpu')
        self.parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        self.parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
        self.parser.add_argument('--test_flop', action='store_true', default=True, help='See utils/tools for usage')

    def parse_args(self, input_args=None):
        self.args = self.parser.parse_args(args=input_args)
        return self.args
    def get_defaults(self):
        return vars(self.parser.parse_args([]))

