import logging
from datetime import datetime
from pathlib import Path

import torch

from rmp.utils.logger import init_fancy_logging
from models import DLinear, LGEANet, Lin, TransformerTSFv2, wLMS, XGBoostTSF, MLP, WaveletLSTM, Many2Many

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#RESULT_DIR = Path(".")  # change
RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(exist_ok=True)
db_root = Path("/mnt/nas-wang/nas-ssd/Scripts/respiratory-signal-database/open_access_rpm_signals_master.db")  # change to the downloaded db-file

DATALAKE = db_root
NUM_WORKERS = 0

SAVED_MODELS_DIR = Path("../trained_models")
LOGGING_FILE = RESULT_DIR / f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}_debug.log"

MODEL_DICT = {
            'DLinear': DLinear,
            'MLP': MLP, # Pure 3 layer LSTM
            'XGBoostTSF': XGBoostTSF,
            'wLMS': wLMS,
            'WaveletLSTM': WaveletLSTM,
            'LGEANet': LGEANet,
            'LSTM': Lin, 
            'TransformerTSFv2':TransformerTSFv2,
            'Many2Many':Many2Many,
        }

init_fancy_logging(
    handlers=[logging.StreamHandler(), logging.FileHandler(LOGGING_FILE)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info("DATALAKE: %s", DATALAKE)
logger.info("RESULT_DIR: %s", RESULT_DIR)
logger.info("DEVICE: %s", DEVICE)
logger.info("NUM_WORKERS: %s", NUM_WORKERS)
logger.info("LOGGING FILE: %s", LOGGING_FILE)
