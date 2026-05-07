import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch
import matplotlib.pyplot as plt
import requests
from IPython.display import display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

RAW_DIR  = "/kaggle/input/datasets/vicentelorenzomarn/dataraw"
WORK_DIR = "/kaggle/working"
Path(f"{WORK_DIR}/models").mkdir(parents=True, exist_ok=True)
Path(f"{WORK_DIR}/results").mkdir(parents=True, exist_ok=True)
print("✅ Imports OK")


FEATURE_COLS = [
    "close",        "volume",       "return_1",     "return_7",
    "rsi_14",       "macd",         "macd_signal",  "bb_position",
    "bb_width",     "atr_14",       "volume_ratio", "hour_sin",
    "hour_cos",     "dow_sin",      "dow_cos",      "fear_greed",
    "return_4h",    "rsi_4h",       "return_1d",    "rsi_1d",
]
N_FEATURES = len(FEATURE_COLS)

SAC_CONFIG = {
    "raw_dir"        : RAW_DIR,
    "data_dir"       : WORK_DIR,
    "work_dir"       : WORK_DIR,
    "coins"          : ["BTC", "ETH", "SOL", "AVAX"],
    "granularity"    : "4h",
    "train_ratio"    : 0.70,
    "val_ratio"      : 0.15,
    "feature_cols"   : FEATURE_COLS,
    "horizons"       : [1, 2, 3, 4],
    "horizon_weights": [1.0, 0.85, 0.70, 0.55],
    # Entorno
    "lookback"       : 32,
    "initial_capital": 10_000.0,
    "commission"     : 0.001,
    "slippage"       : 0.0005,
    "max_dd_limit"   : 0.25,
    # Reward shaping
    "reward_alpha"   : 1.0,
    "reward_beta"    : 0.5,
    "reward_gamma"   : 0.3,
    "reward_delta"   : 1.0,
    "reward_epsilon" : 0.1,
    # SAC — FIX: eliminadas claves duplicadas (raw_dir y learning_starts)
    "learning_rate"  : 1e-4,
    "learning_starts": 2_000,
    "buffer_size"    : 100_000,
    "batch_size"     : 256,
    "tau"            : 0.005,
    "gamma"          : 0.99,
    "ent_coef"       : "auto",
    "policy_kwargs"  : dict(log_std_init=-3, net_arch=[256, 256]),
    "net_arch"       : [256, 256],
    # Entrenamiento
    "total_timesteps": 300_000,
    "eval_freq"      : 5_000,
}

trained_models = {}
for coin in SAC_CONFIG["coins"]:
    best_path = f"{WORK_DIR}/models/sac_{coin}_best.zip"
    if Path(best_path).exists():
        trained_models[coin] = SAC.load(
            best_path,
            env    = envs[coin]["train_vec"],
            device = "auto",
        )
        print(f"  [{coin}] ✅ cargado desde {best_path}")
    else:
        print(f"  [{coin}] ⚠️  no encontrado")