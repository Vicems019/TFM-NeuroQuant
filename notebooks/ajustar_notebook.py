import json, copy, sys
from pathlib import Path

if len(sys.argv) > 1:
    input_path = Path(sys.argv[1])
else:
    input_path = Path(__file__).parent / "crypto-lstms.ipynb"
    if not input_path.exists():
        print(f" ERROR: No se encuentra {input_path}")
        sys.exit(1)
    if input_path.is_dir():
        print(f" ERROR: La ruta apunta a una carpeta, no a un archivo: {input_path}")
        sys.exit(1)

output_path = input_path.parent / "crypto-lstm_v15.ipynb"
print(f"Procesando: {input_path.name}")
print(f"Resultado: {output_path.name}")

with open(input_path, encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src}

# ── CELL 1: add PREPROCESSED_DIR + ZoneInfo ───────────────────────────────────
nb['cells'][1]['source'] = (
    '!pip install torch pandas numpy scikit-learn matplotlib statsmodels --quiet\n'
    '\n'
    'import os, math, time, pickle, warnings\n'
    'import numpy as np\n'
    'import pandas as pd\n'
    'import matplotlib.pyplot as plt\n'
    'import matplotlib.gridspec as gridspec\n'
    'from sklearn.preprocessing import RobustScaler\n'
    'from sklearn.metrics import mean_absolute_error, mean_squared_error\n'
    'from statsmodels.tsa.stattools import adfuller, acf, pacf\n'
    'from statsmodels.tsa.seasonal import seasonal_decompose\n'
    'import torch\n'
    'import torch.nn as nn\n'
    'from torch.utils.data import Dataset, DataLoader\n'
    'from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts\n'
    'import requests\n'
    'from datetime import datetime, timezone\n'
    'from zoneinfo import ZoneInfo\n'
    '\n'
    'warnings.filterwarnings("ignore")\n'
    '\n'
    'DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
    'print(f"Dispositivo: {DEVICE}")\n'
    'if torch.cuda.is_available():\n'
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
    '\n'
    'SEED = 42\n'
    'torch.manual_seed(SEED)\n'
    'np.random.seed(SEED)\n'
    'if torch.cuda.is_available():\n'
    '    torch.cuda.manual_seed_all(SEED)\n'
    '\n'
    'RAW_DIR          = "/kaggle/input/datasets/vicentelorenzomarn/dataraw"\n'
    'PREPROCESSED_DIR = "/kaggle/input/datasets/vicentelorenzomarn/preprocessed"\n'
    'WORK_DIR         = "/kaggle/working"\n'
    'MODELS_DIR       = "/kaggle/input/datasets/vicentelorenzomarn/crypto-models"\n'
    'TZ_MADRID        = ZoneInfo("Europe/Madrid")\n'
    'os.makedirs(WORK_DIR, exist_ok=True)\n'
    'os.makedirs("plots", exist_ok=True)\n'
    '\n'
    'MAPE_HISTORICO = {\n'
    '    "BTC":  0.0055, "ETH":  0.0091, "SOL":  0.0101,\n'
    '    "AVAX": 0.0106,\n'
    '}\n'
    'UMBRALES_BT = {\n'
    '    "BTC": 0.003, "ETH": 0.003, "SOL": 0.003,\n'
    '    "AVAX": 0.003,\n'
    '}\n'
)

# ── CELL 3: fix assert 44→37, data_dir→PREPROCESSED_DIR ──────────────────────
src3 = ''.join(nb['cells'][3]['source'])
src3 = src3.replace(
    'assert len(FEATURE_COLS) == 44, f"Se esperaban 44 features, hay {len(FEATURE_COLS)}"',
    'assert len(FEATURE_COLS) == 37, f"Se esperaban 37 features, hay {len(FEATURE_COLS)}"'
)
src3 = src3.replace(
    '"data_dir":        WORK_DIR,',
    '"data_dir":        PREPROCESSED_DIR,'
)
nb['cells'][3]['source'] = src3

# ── CELL 12: fix hardcoded 44→37 in test print ────────────────────────────────
src12 = ''.join(nb['cells'][12]['source'])
src12 = src12.replace(
    'model_test = CryptoLSTM(44, 4, cfg_btc)',
    'model_test = CryptoLSTM(37, 4, cfg_btc)'
).replace(
    'print(f"  CryptoLSTM (44 features): {count_params(model_test):>10,} parámetros")',
    'print(f"  CryptoLSTM (37 features): {count_params(model_test):>10,} parámetros")'
).replace(
    'print("Estimación de parámetros (BTC, 37 features, seq=24):")',
    'print("Estimación de parámetros (BTC, 37 features, seq=24):")'
)
nb['cells'][12]['source'] = src12

# ── NEW CELL (inference support) — insert before cell 24 ─────────────────────
inference_support = (
    'COINS_MAP = {\n'
    '    "BTC": "BTCUSDT", "ETH": "ETHUSDT",\n'
    '    "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",\n'
    '}\n'
    'BINANCE_URLS = [\n'
    '    "https://api.binance.us/api/v3/klines",\n'
    '    "https://api.binance.com/api/v3/klines",\n'
    ']\n'
    '\n'
    'def _binance_get(params, timeout=15):\n'
    '    for url in BINANCE_URLS:\n'
    '        try:\n'
    '            r = requests.get(url, params=params, timeout=timeout)\n'
    '            r.raise_for_status()\n'
    '            return r.json()\n'
    '        except Exception:\n'
    '            continue\n'
    '    return None\n'
    '\n'
    'def download_recent(symbol, interval, limit=500):\n'
    '    batch = _binance_get({"symbol": symbol, "interval": interval, "limit": limit})\n'
    '    if not batch:\n'
    '        return None\n'
    '    rows = [{\n'
    '        "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),\n'
    '        "open":   float(c[1]), "high":  float(c[2]),\n'
    '        "low":    float(c[3]), "close": float(c[4]),\n'
    '        "volume": float(c[5]),\n'
    '    } for c in batch]\n'
    '    df = (pd.DataFrame(rows)\n'
    '            .drop_duplicates("timestamp")\n'
    '            .sort_values("timestamp")\n'
    '            .reset_index(drop=True))\n'
    '    vela_actual = pd.Timestamp.now(tz="UTC").floor("1h")\n'
    '    df = df[df["timestamp"] < vela_actual].reset_index(drop=True)\n'
    '    return df\n'
    '\n'
    'def download_fear_greed(limit=10):\n'
    '    try:\n'
    '        resp = requests.get(\n'
    '            f"https://api.alternative.me/fng/?limit={limit}&format=json", timeout=15\n'
    '        )\n'
    '        resp.raise_for_status()\n'
    '        data = resp.json()["data"]\n'
    '        df = pd.DataFrame([{\n'
    '            "date": (pd.to_datetime(int(d["timestamp"]), unit="s", utc=True)\n'
    '                     .tz_convert("Europe/Madrid").normalize().tz_localize(None)),\n'
    '            "fear_greed": int(d["value"]),\n'
    '        } for d in data]).sort_values("date").reset_index(drop=True)\n'
    '        return df\n'
    '    except Exception:\n'
    '        dates = (pd.date_range("2020-01-01", periods=30, freq="D", tz="Europe/Madrid")\n'
    '                   .tz_localize(None))\n'
    '        return pd.DataFrame({"date": dates, "fear_greed": 50})\n'
    '\n'
    'def _calc_indicators_inf(df):\n'
    '    df = df.copy()\n'
    '    df["ema_7"]  = df["close"].ewm(span=7,  adjust=False).mean()\n'
    '    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()\n'
    '    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()\n'
    '    delta = df["close"].diff()\n'
    '    ag14  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()\n'
    '    al14  = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()\n'
    '    df["rsi_14"] = 100 - 100 / (1 + ag14 / (al14 + 1e-9))\n'
    '    ag6  = delta.clip(lower=0).ewm(alpha=1/6, min_periods=6, adjust=False).mean()\n'
    '    al6  = (-delta).clip(lower=0).ewm(alpha=1/6, min_periods=6, adjust=False).mean()\n'
    '    df["rsi_6"]  = 100 - 100 / (1 + ag6  / (al6  + 1e-9))\n'
    '    ema12 = df["close"].ewm(span=12, adjust=False).mean()\n'
    '    ema26 = df["close"].ewm(span=26, adjust=False).mean()\n'
    '    df["macd"]        = ema12 - ema26\n'
    '    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()\n'
    '    df["macd_hist"]   = df["macd"] - df["macd_signal"]\n'
    '    sma20 = df["close"].rolling(20).mean()\n'
    '    std20 = df["close"].rolling(20).std()\n'
    '    bb_u  = sma20 + 2 * std20\n'
    '    bb_l  = sma20 - 2 * std20\n'
    '    df["bb_width"]    = (bb_u - bb_l) / (sma20 + 1e-9)\n'
    '    df["bb_position"] = (df["close"] - bb_l) / (bb_u - bb_l + 1e-9)\n'
    '    hl = df["high"] - df["low"]\n'
    '    hc = (df["high"] - df["close"].shift(1)).abs()\n'
    '    lc = (df["low"]  - df["close"].shift(1)).abs()\n'
    '    df["atr_14"] = (pd.concat([hl, hc, lc], axis=1).max(axis=1)\n'
    '                    .ewm(alpha=1/14, min_periods=14, adjust=False).mean())\n'
    '    return df\n'
    '\n'
    'def _build_features_inf(df_1h):\n'
    '    df = df_1h.copy()\n'
    '    if df["timestamp"].dt.tz is None:\n'
    '        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")\n'
    '    df = _calc_indicators_inf(df)\n'
    '    df["close_vs_ema7"]  = df["close"] / (df["ema_7"]  + 1e-9) - 1\n'
    '    df["close_vs_ema14"] = df["close"] / (df["ema_14"] + 1e-9) - 1\n'
    '    df["close_vs_ema50"] = df["close"] / (df["ema_50"] + 1e-9) - 1\n'
    '    df["return_1"]  = df["close"].pct_change(1)\n'
    '    df["return_4"]  = df["close"].pct_change(4)\n'
    '    df["return_24"] = df["close"].pct_change(24)\n'
    '    for lag in [1, 2, 3, 4]:\n'
    '        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)\n'
    '    df["atr_14_norm"]     = df["atr_14"] / (df["close"] + 1e-9)\n'
    '    df["volume_ratio"]    = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9)\n'
    '    df["volume_ratio_4h"] = df["volume"] / (df["volume"].rolling(4).mean()  + 1e-9)\n'
    '    h24 = df["high"].rolling(24).max()\n'
    '    l24 = df["low"].rolling(24).min()\n'
    '    df["range_pos_24h"]    = (df["close"] - l24)  / (h24 - l24 + 1e-9)\n'
    '    df["dist_to_high_24h"] = (h24 - df["close"])  / (df["close"] + 1e-9)\n'
    '    df["dist_to_low_24h"]  = (df["close"] - l24)  / (df["close"] + 1e-9)\n'
    '    h7d = df["high"].rolling(7*24).max()\n'
    '    l7d = df["low"].rolling(7*24).min()\n'
    '    df["range_pos_7d"]  = (df["close"] - l7d) / (h7d - l7d + 1e-9)\n'
    '    ret = df["close"].pct_change()\n'
    '    df["volatility_24h"] = ret.rolling(24).std()\n'
    '    df["vol_ratio"]      = df["volatility_24h"] / (ret.rolling(7*24).std() + 1e-9)\n'
    '    ts_mad = df["timestamp"].dt.tz_convert("Europe/Madrid")\n'
    '    df["hour_sin"]  = np.sin(2 * np.pi * ts_mad.dt.hour      / 24)\n'
    '    df["hour_cos"]  = np.cos(2 * np.pi * ts_mad.dt.hour      / 24)\n'
    '    df["dow_sin"]   = np.sin(2 * np.pi * ts_mad.dt.dayofweek / 7)\n'
    '    df["dow_cos"]   = np.cos(2 * np.pi * ts_mad.dt.dayofweek / 7)\n'
    '    df["month_sin"] = np.sin(2 * np.pi * ts_mad.dt.month     / 12)\n'
    '    df["month_cos"] = np.cos(2 * np.pi * ts_mad.dt.month     / 12)\n'
    '    return df\n'
    '\n'
    'def _merge_fg_inf(df_1h, fg_df):\n'
    '    df = df_1h.copy()\n'
    '    df["date"] = (df["timestamp"]\n'
    '                  .dt.tz_convert("Europe/Madrid")\n'
    '                  .dt.normalize()\n'
    '                  .dt.tz_localize(None))\n'
    '    df = df.merge(fg_df[["date", "fear_greed"]], on="date", how="left")\n'
    '    df["fear_greed"] = df["fear_greed"].ffill().fillna(50)\n'
    '    df.drop(columns=["date"], inplace=True)\n'
    '    return df\n'
    '\n'
    'def _merge_multitf_inf(df_1h, df_4h_raw, df_1d_raw):\n'
    '    def prep(raw, col_name, shift):\n'
    '        d = _calc_indicators_inf(raw.copy())\n'
    '        if d["timestamp"].dt.tz is None:\n'
    '            d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")\n'
    '        d[col_name] = d["close"].pct_change(1)\n'
    '        out = d[["timestamp", "rsi_14", col_name]].copy()\n'
    '        out.columns = ["timestamp", f"rsi_{col_name.split(\"_\")[1]}", col_name]\n'
    '        out["timestamp"] += shift\n'
    '        return out\n'
    '    f4h = prep(df_4h_raw, "return_4h", pd.Timedelta(hours=4))\n'
    '    f1d = prep(df_1d_raw, "return_1d", pd.Timedelta(days=1))\n'
    '    df = pd.merge_asof(df_1h.sort_values("timestamp"),\n'
    '                       f4h.sort_values("timestamp"), on="timestamp", direction="backward")\n'
    '    df = pd.merge_asof(df, f1d.sort_values("timestamp"),\n'
    '                       on="timestamp", direction="backward")\n'
    '    return df\n'
    '\n'
    'print("✅ Funciones de inferencia cargadas.")\n'
)

# Find position of current cell 24 and insert new cell before it
cell24_pos = 24
nb['cells'].insert(cell24_pos, code_cell(inference_support))

# ── CELL 25 (was 24): rewrite get_recent_data ─────────────────────────────────
new_cell24 = (
    'def get_recent_data(coin, cfg):\n'
    '    seq_len = cfg["seq_len"]\n'
    '    buffer  = max(7 * 24 + 26, seq_len + 60)\n'
    '    symbol  = COINS_MAP[coin]\n'
    '\n'
    '    df_1h = download_recent(symbol, "1h", buffer)\n'
    '    if df_1h is None or len(df_1h) < seq_len:\n'
    '        raise RuntimeError(f"{coin}: no se pudo descargar datos 1h (disponibles: {len(df_1h) if df_1h is not None else 0})")\n'
    '\n'
    '    df = _build_features_inf(df_1h)\n'
    '\n'
    '    fg = download_fear_greed(limit=10)\n'
    '    df = _merge_fg_inf(df, fg)\n'
    '\n'
    '    df_4h = download_recent(symbol, "4h", 100)\n'
    '    df_1d = download_recent(symbol, "1d", 30)\n'
    '    if df_4h is not None and df_1d is not None:\n'
    '        df = _merge_multitf_inf(df, df_4h, df_1d)\n'
    '    else:\n'
    '        for col in ["rsi_4h", "return_4h", "rsi_1d", "return_1d"]:\n'
    '            df[col] = 0.0\n'
    '\n'
    '    return df.dropna().reset_index(drop=True)\n'
    '\n'
    '\n'
    'def predict_next(coin, result):\n'
    '    model       = result["model"]\n'
    '    feat_scaler = result["feat_scaler"]\n'
    '    reg_scaler  = result["reg_scaler"]\n'
    '    cfg         = result["cfg"]\n'
    '    feat_cols   = result["feat_cols"]\n'
    '    seq_len     = cfg["seq_len"]\n'
    '    mape        = MAPE_HISTORICO.get(coin, 0.01)\n'
    '\n'
    '    df_recent = get_recent_data(coin, cfg)\n'
    '    if len(df_recent) < seq_len:\n'
    '        raise ValueError(f"Datos insuficientes: {len(df_recent)} < {seq_len}")\n'
    '\n'
    '    available = [c for c in feat_cols if c in df_recent.columns]\n'
    '    missing   = [c for c in feat_cols if c not in df_recent.columns]\n'
    '    if missing:\n'
    '        print(f"  ⚠ {coin}: features ausentes en inferencia: {missing}")\n'
    '\n'
    '    X_raw      = df_recent[available].iloc[-seq_len:].values.astype(np.float32)\n'
    '    X_scaled   = feat_scaler.transform(X_raw)\n'
    '    X_tensor   = torch.tensor(X_scaled[np.newaxis], dtype=torch.float32).to(DEVICE)\n'
    '\n'
    '    last_close = float(df_recent["close"].iloc[-1])\n'
    '    last_time  = df_recent["timestamp"].iloc[-1].tz_convert("Europe/Madrid")\n'
    '\n'
    '    model.eval()\n'
    '    with torch.no_grad():\n'
    '        reg_vals = reg_scaler.inverse_transform(model(X_tensor).cpu().numpy())[0]\n'
    '\n'
    '    preds = {}\n'
    '    for i, h in enumerate(cfg["horizons"]):\n'
    '        ret    = float(reg_vals[i])\n'
    '        price  = round(last_close * (1 + ret), 4)\n'
    '        margen = price * mape\n'
    '        preds[f"{h}h"] = {\n'
    '            "precio_estimado":    price,\n'
    '            "precio_min":         round(price - margen, 4),\n'
    '            "precio_max":         round(price + margen, 4),\n'
    '            "retorno_esperado_%": round(ret * 100, 3),\n'
    '            "direccion":          "↑ SUBE" if ret > 0 else "↓ BAJA",\n'
    '        }\n'
    '    return last_close, last_time, preds\n'
    '\n'
    '\n'
    'print("=" * 62)\n'
    'print("  PREDICCIONES EN TIEMPO REAL")\n'
    'print("=" * 62)\n'
    'all_predictions = {}\n'
    'for coin, res in all_results.items():\n'
    '    try:\n'
    '        last_close, last_time, preds = predict_next(coin, res)\n'
    '        all_predictions[coin] = preds\n'
    '        print(f"\\n  {coin} | Cierre: {last_close:.4f} USD | {last_time.strftime(\'%Y-%m-%d %H:%M %Z\')}")\n'
    '        print(f"  {\'─\'*60}")\n'
    '        print(f"  {\'Horizonte\':>10} | {\'Estimado\':>12} | {\'Min\':>12} | {\'Max\':>12} | Dir")\n'
    '        print(f"  {\'─\'*10}-+-{\'─\'*12}-+-{\'─\'*12}-+-{\'─\'*12}-+----")\n'
    '        for h, p in preds.items():\n'
    '            arrow = "↑" if p["direccion"] == "↑ SUBE" else "↓"\n'
    '            print(f"  {h:>10} | {p[\'precio_estimado\']:>12.4f} | "\n'
    '                  f"{p[\'precio_min\']:>12.4f} | {p[\'precio_max\']:>12.4f} | {arrow} "\n'
    '                  f"{p[\'retorno_esperado_%\']:+.3f}%")\n'
    '    except Exception as e:\n'
    '        print(f"  ✗ {coin}: {e}")\n'
    '        import traceback; traceback.print_exc()\n'
)

nb['cells'][25]['source'] = new_cell24

print(f"✅ Notebook guardado con {len(nb['cells'])} celdas")

# Verify key changes
# ─── GUARDAR NOTEBOOK MODIFICADO ──────────────────────────────────────────────
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print(f"✅ Notebook guardado en {output_path}")

# ─── VERIFICACIONES RÁPIDAS ───────────────────────────────────────────────────
with open(output_path, encoding='utf-8') as f:
    nb2 = json.load(f)

checks = {
    "PREPROCESSED_DIR en cell 1":     "PREPROCESSED_DIR" in ''.join(nb2['cells'][1]['source']),
    "ZoneInfo en cell 1":             "ZoneInfo" in ''.join(nb2['cells'][1]['source']),
    "assert 37 en cell 3":            "== 37" in ''.join(nb2['cells'][3]['source']),
    "data_dir=PREPROCESSED en cell 3":"PREPROCESSED_DIR" in ''.join(nb2['cells'][3]['source']),
    "download_recent en cell 24":     "def download_recent" in ''.join(nb2['cells'][24]['source']),
    "vela_actual filter":             "vela_actual" in ''.join(nb2['cells'][24]['source']),
    "Madrid tz en inferencia":        "Europe/Madrid" in ''.join(nb2['cells'][24]['source']),
    "get_recent_data en cell 25":     "def get_recent_data" in ''.join(nb2['cells'][25]['source']),
    "feat_scaler.transform en cell25":"feat_scaler.transform" in ''.join(nb2['cells'][25]['source']),
}
for k, v in checks.items():
    print(f"  {'✅' if v else '✗'} {k}")