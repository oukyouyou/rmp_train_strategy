"""For each model, a hyperparameter search space is defined."""

import numpy as np
from hyperopt import hp

SEARCH_SPACE_LINEAR = {
    "seq_len": hp.choice("seq_len", 
                                [4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 
                                 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]),
    "alpha": hp.choice(
        "alpha",
        [
            1 * 1e-05,
            1 * 10e-04,
            1 * 10e-03,
            1 * 10e-02,
            1 * 10e-01,
            0,
            0.5,
            1,
            1.5,
            2,
            10,
        ],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [64, 128, 256, 512]
    ),
}
SEARCH_SPACE_LSTM = { 
    # "DMS_flag": hp.choice("DMS_flag",[True, False]),
    "seq_len": hp.choice("seq_len", [4, 8, 16, 32, 64, 128]), 
    "num_layers": hp.choice("num_layers", [1, 2, 3, 4, 8]),
    "hidden_dim": hp.choice("hidden_dim", [8, 32, 64, 128, 256]),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [64, 128, 256, 512]
    ),
}


SEARCH_SPACE_LSTM_batch_1 = { 
    # "DMS_flag": hp.choice("DMS_flag",[True, False]),
    "seq_len": hp.choice("seq_len", [4, 8, 16, 32, 64, 128]), 
    "num_layers": hp.choice("num_layers", [1, 2, 3, 4, 8]),
    "hidden_dim": hp.choice("hidden_dim", [8, 32, 64, 128, 256]),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [1]
    ),
}


SEARCH_SPACE_MAMBA = {
    "seq_len": hp.choice("seq_len", [16, 32, 64, 128, 256, 384, 512]),  # lookback/input length
    "d_model": hp.choice("d_model", [32, 64, 128, 256]),  # Mamba embedding dimension
    "num_layers": hp.choice("num_layers", [1, 2, 3, 4, 6]),  # number of stacked Mamba layers
    "d_state": hp.choice("d_state", [8, 16, 32, 64]),  # state size in SSM
    "d_conv": hp.choice("d_conv", [3, 5, 7]),  # kernel size of convolution
    "expand": hp.choice("expand", [1, 2, 4]),  # expansion ratio for intermediate dims
    "learning_rate": hp.choice("learning_rate", [1e-3, 5e-4, 1e-4]),  # optimizer lr
    "batch_size": hp.choice("batch_size", [64, 128, 256, 512]),  # batch size
}

SEARCH_SPACE_TIMESNET = {
    # "norm_flag": hp.choice("norm_flag",[True, False]),
    "seq_len": hp.choice("seq_len", [16, 32, 64, 128, 256, 384, 512]),  # lookback/input length
    "d_model": hp.choice("d_model", [16, 32, 64]),  # 投影维度
    "num_layers": hp.choice("num_layers", [1, 2, 3]),  # TimesBlock 层数
    "top_k": hp.choice("top_k", [1, 2, 3]),  # FFT中选取的周期数
    "d_ff": hp.choice("d_ff", [8, 16, 32]),  # Inception中间维度 （FF_d）
    "num_kernels": hp.choice("num_kernels", [3, 5, 7]),  # Inception中的卷积核数量（对应不同kernel size）
    "learning_rate": hp.choice("learning_rate", [1e-3, 5e-4, 1e-4]),  # 学习率
    "batch_size": hp.choice("batch_size", [64, 128, 256, 512]),  # 批大小
}

SEARCH_SPACE_MLP = {
    #"seq_len": hp.choice("seq_len", [4, 16, 32, 64, 128, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024]),
    "seq_len": hp.choice("seq_len", [4, 16, 32, 64, 128, 256, 384, 512, 640, 768,  896, 1024]),
    "num_layers": hp.choice("num_layers", [2, 3, 4, 8]),
    "hidden_dim": hp.choice("hidden_dim", [8, 32, 64, 128, 256, 512]),
    # "activation_function": hp.choice(
    #     "activation_function",['relu', 'leaky_relu', 'tanh', 'sigmoid', 'Dyt']
    # ),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [ 128, 256, 512]
    ),
}
SEARCH_SPACE_TRANSFORMER = {
    "seq_len": hp.choice("seq_len", [1, 4, 8, 16, 32, 64]),
    "num_layers": hp.choice("num_layers", [1, 4, 8]),
    "n_heads": hp.choice("n_heads", [1, 4, 8, 16]),
    "embedding_dim": hp.choice("embedding_dim", [8, 32, 64, 128]),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [64, 128, 256, 512]
    ),
}

SEARCH_SPACE_TRANSFORMER_TSFv2 = {
    "seq_len": hp.quniform("seq_len", 25, 100, 5), 
    "num_layers": hp.choice("num_layers", [1, 2, 4, 8]),
    "n_heads": hp.choice("n_heads", [1, 4, 8]),
    "layer_dim_val": hp.choice("layer_dim_val", [8, 32, 64]),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [64, 128, 256, 512])

}

SEARCH_SPACE_DLINEAR = {
    "seq_len": hp.choice(
        "seq_len",
         [4, 8, 16, 32, 64, 128, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024],
    ),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
    "batch_size": hp.choice(
        "batch_size", [64, 128, 256, 512]
    ),
}
SEARCH_SPACE_XGB = {
    "seq_len": hp.quniform("seq_len", 10, 500, 20),
    "subsample_baselearner": hp.choice("subsample_baselearner", [0.95]),
    "max_depth": hp.quniform("max_depth", 3, 18, 1),
    "gamma": hp.uniform("gamma", 0, 4),
    "min_child_weight": hp.quniform("min_child_weight", 2, 10, 1),
    "n_estimators": hp.quniform("n_estimators", 20, 500, 10),
    "reg_lambda": hp.loguniform("reg_lambda", np.log(1), np.log(100)),
    "learning_rate": hp.choice(
        "learning_rate",
        [10e-03,5 * 10e-04, 10e-04], 
    ),
}
SEARCH_SPACE_ARIMA = {
    "seq_len": hp.choice(
        "seq_len",
         [4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
    ),

}

