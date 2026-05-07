import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import random
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

BASE_PRICES = {"BTC": 78448, "ETH": 3120, "SOL": 148, "XRP": 0.512, "AVAX": 28.4}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Features usadas en el modelo (de crypto-lstm.ipynb)
FEATURE_COLS = [
    "close","volume","return_1","return_7",
    "rsi_14","macd","macd_signal","bb_position",
    "bb_width","atr_14","volume_ratio","hour_sin",
    "hour_cos","dow_sin","dow_cos","fear_greed",
    "return_4h","rsi_4h","return_1d","rsi_1d", "volatility_7"
]
# Configuraciones por moneda (del cuaderno crypto-lstm.ipynb)
COIN_CONFIGS = {
    "BTC": {
        "lstm_hidden": 256, "lstm_layers": 2,
        "lstm_dropout": 0.145, "head_dropout": 0.275,
        "head_hidden": 64, "seq_len": 24,
    },
    "ETH": {
        "lstm_hidden": 64, "lstm_layers": 3,
        "lstm_dropout": 0.134, "head_dropout": 0.284,
        "head_hidden": 64, "seq_len": 24,
    },
    "SOL": {
        "lstm_hidden": 256, "lstm_layers": 2,
        "lstm_dropout": 0.29, "head_dropout": 0.24,
        "head_hidden": 64, "seq_len": 24,
    },
    "AVAX": {
        "lstm_hidden": 128, "lstm_layers": 2,
        "lstm_dropout": 0.32, "head_dropout": 0.13,
        "head_hidden": 32, "seq_len": 48,
    },
}
N_FEATURES = len(FEATURE_COLS)
N_HORIZONS = 4
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "lstm"

# Cache para no recargar el modelo en cada llamada
_MODEL_CACHE: dict = {}


class CryptoLSTM(nn.Module):
    def __init__(self, n_features: int, n_horizons: int, cfg: dict):
        super().__init__()
        H      = cfg["lstm_hidden"]
        head_h = cfg["head_hidden"]
        self.input_proj = nn.Sequential(nn.Linear(n_features, H), nn.LayerNorm(H))
        lstm_drop = cfg["lstm_dropout"] if cfg["lstm_layers"] > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=H, hidden_size=H,
            num_layers=cfg["lstm_layers"], dropout=lstm_drop, batch_first=True,
        )
        self.attn = nn.Linear(H, 1)
        self.norm = nn.LayerNorm(H)
        self.drop = nn.Dropout(cfg["head_dropout"])
        self.mlp  = nn.Sequential(
            nn.Linear(H, head_h), nn.LayerNorm(head_h), nn.GELU(),
            nn.Dropout(cfg["head_dropout"]),
            nn.Linear(head_h, head_h // 2), nn.GELU(),
        )
        self.head = nn.Linear(head_h // 2, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x           = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        attn_w      = torch.softmax(self.attn(lstm_out), dim=1)
        context     = (attn_w * lstm_out).sum(dim=1)
        context     = self.drop(self.norm(context))
        return self.head(self.mlp(context))


def _load_model(cripto: str) -> nn.Module:
    """Carga el .pt una sola vez y lo cachea en memoria."""
    cripto = cripto.upper()
    if cripto in _MODEL_CACHE:
        return _MODEL_CACHE[cripto]

    # Buscar el .pt — acepta cualquier fold (best_BTC_fold5.pt, etc.)
    candidates = list(MODELS_DIR.glob(f"*{cripto}*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún .pt para {cripto} en {MODELS_DIR}"
        )
    # Si hay varios folds, coger el de mayor número
    pt_path = sorted(candidates)[-1]

    cfg   = COIN_CONFIGS[cripto]
    model = CryptoLSTM(N_FEATURES, N_HORIZONS, cfg)
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    _MODEL_CACHE[cripto] = model
    print(f"  [{cripto}] ✅ modelo cargado desde {pt_path.name}")
    return model


def _build_sequences(df, seq_len: int):
    feat = df[FEATURE_COLS].values.astype(np.float32)
    mu   = feat.mean(axis=0)
    std  = feat.std(axis=0) + 1e-8
    feat = np.clip((feat - mu) / std, -5., 5.)

    closes = df["close"].values
    n      = len(closes)

    max_h = 4
    targets = np.stack([
        (closes[h : n - max_h + h] - closes[:n - max_h]) /
        (closes[:n - max_h] + 1e-8)
        for h in [1, 2, 3, 4]
    ], axis=1)  # (n - max_h, 4)

    X, y = [], []
    max_i = len(targets) - seq_len
    for i in range(max_i):
        X.append(feat[i : i + seq_len])
        y.append(targets[i + seq_len - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def get_metrica(cripto: str, df_test=None) -> dict:
    """
    Calcula métricas del modelo LSTM sobre el test set.

    Args:
        cripto  : "BTC", "ETH", "SOL" o "AVAX"
        df_test : DataFrame del test set. Si es None, busca en data/processed/.

    Returns:
        dict con rmse_lstm, mae_lstm, r2_lstm, accuracy_lstm
    """
    cripto = cripto.upper()
    cfg    = COIN_CONFIGS[cripto]
    model  = _load_model(cripto)

    print(f"Features usadas para entrenar: {len(FEATURE_COLS)}")
    print(FEATURE_COLS)


    # Cargar datos si no se pasan directamente
    if df_test is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "preprocessed" / f"{cripto}_hourly.csv"
        import pandas as pd
        df_full  = pd.read_csv(data_path, parse_dates=["timestamp"])
        n        = len(df_full)
        n_train  = int(n * 0.70)
        n_val    = int(n * 0.15)
        df_test  = df_full.iloc[n_train + n_val:].reset_index(drop=True)

    X, y_true = _build_sequences(df_test, cfg["seq_len"])

    # Inferencia sin gradientes
    with torch.no_grad():
        preds = model(torch.from_numpy(X)).numpy()  # (N, 4)

    # Usar solo horizonte h=1 para las métricas escalares
    y1_true = y_true[:, 0]
    y1_pred = preds[:, 0]

    rmse     = float(np.sqrt(mean_squared_error(y1_true, y1_pred)))
    mae      = float(mean_absolute_error(y1_true, y1_pred))
    r2       = float(r2_score(y1_true, y1_pred))

    # Directional accuracy: acierto en la dirección del movimiento
    dir_true = np.sign(y1_true)
    dir_pred = np.sign(y1_pred)
    accuracy = float((dir_true == dir_pred).mean())

    return {
        "rmse_lstm"    : rmse,
        "mae_lstm"     : mae,
        "r2_lstm"      : r2,
        "accuracy_lstm": accuracy,
    }