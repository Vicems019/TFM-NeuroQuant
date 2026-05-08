import os, json, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# -- Rutas --------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")
KAGGLE_ID = "vicentelorenzomarn/dataraw"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# -- Config -------------------------------------------------------------------
COINS_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "AVAX": "AVAXUSDT",
}
START_DATES = {
    "BTC": "2020-01-01", "ETH": "2020-01-01", "SOL": "2020-09-01",
    "XRP": "2020-01-01", "AVAX": "2020-10-01",
}
INTERVALS   = ["1h", "4h", "1d"]
BINANCE_URLS = [
    "https://api.binance.us/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]

# -- Feature Engineering Config -----------------------------------------------
FEATURE_COLS = [
    "close",        "volume",       "return_1",     "return_7",
    "rsi_14",       "macd",         "macd_signal",  "bb_position",
    "bb_width",     "atr_14",       "volume_ratio", "hour_sin",
    "hour_cos",     "dow_sin",      "dow_cos",      "fear_greed",
    "return_4h",    "rsi_4h",       "return_1d",    "rsi_1d", 
]
HORIZONS = [1, 2, 3, 4]

# -- Binance ------------------------------------------------------------------
def binance_get(params, timeout=15):
    for url in BINANCE_URLS:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    return None

def _date_to_ms(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)

def download_historical(symbol, interval, start_date):
    """Descarga historico completo desde start_date (solo primera vez)."""
    start_ms = _date_to_ms(date_str=start_date)
    end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    rows = []
    while start_ms < end_ms:
        batch = binance_get({"symbol": symbol, "interval": interval,
                              "startTime": start_ms, "limit": 1000})
        if not batch:
            break
        for c in batch:
            rows.append({
                "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                "open": float(c[1]), "high": float(c[2]),
                "low":  float(c[3]), "close": float(c[4]),
                "volume": float(c[5]),
            })
        start_ms = batch[-1][0] + 1
        time.sleep(0.15)
    if not rows:
        return None
    return (pd.DataFrame(rows)
              .drop_duplicates("timestamp")
              .sort_values("timestamp")
              .reset_index(drop=True))

def update_csv(path, symbol, interval, coin):
    """
    Siempre descarga solo las ultimas 1000 velas de Binance.
    - Si el CSV existe: añade solo las velas nuevas.
    - Si no existe:     crea el CSV con esas 1000 velas.
    """
    batch = binance_get({"symbol": symbol, "interval": interval, "limit": 1000})
    if batch is None:
        print(f"  {coin} {interval}: sin respuesta de Binance !")
        return

    df_new = pd.DataFrame([{
        "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
        "open":   float(c[1]), "high": float(c[2]),
        "low":    float(c[3]), "close": float(c[4]),
        "volume": float(c[5]), "coin": coin,
    } for c in batch])

    if os.path.exists(path):
        df_existing = pd.read_csv(path, parse_dates=["timestamp"])
        if df_existing["timestamp"].dt.tz is None:
            df_existing["timestamp"] = df_existing["timestamp"].dt.tz_localize("UTC")

        last_ts  = df_existing["timestamp"].max()
        df_new   = df_new[df_new["timestamp"] > last_ts]

        if df_new.empty:
            print(f"  {coin} {interval}: ya al dia ({last_ts.date()}) OK")
            return

        df_out = (pd.concat([df_existing, df_new], ignore_index=True)
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        df_out.to_csv(path, index=False)
        print(f"  {coin} {interval}: +{len(df_new)} velas -> {df_out['timestamp'].max().date()} OK")
    else:
        df_new.to_csv(path, index=False)
        print(f"  {coin} {interval}: creado con {len(df_new)} velas -> {df_new['timestamp'].max().date()} OK")


def update_local():
    print("\n-- Actualizando data/raw/ ----------------------------------")
    for coin, symbol in COINS_MAP.items():
        for interval in ["1h", "4h", "1d"]:
            path = os.path.join(RAW_DIR, f"{coin}_{interval}_raw.csv")
            update_csv(path, symbol, interval, coin)
    print("OK data/raw/ actualizado\n")

# -- Preprocesamiento (desde sac_agent.ipynb) ----------------------------------

def download_fear_greed(limit=2000, verbose=True):
    try:
        resp = requests.get(f"https://api.alternative.me/fng/?limit={limit}&format=json", timeout=15)
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame([
            {"date": pd.to_datetime(int(d["timestamp"]), unit="s", utc=True).normalize(),
             "fear_greed": int(d["value"])}
            for d in data
        ]).sort_values("date").reset_index(drop=True)
        if verbose: print(f"  Fear & Greed: {len(df)} dias OK")
        return df
    except Exception as e:
        if verbose: print(f"  ! Fear & Greed no disponible ({e}) -- usando 50")
        dates = pd.date_range("2020-01-01", periods=2500, freq="D", tz="UTC")
        return pd.DataFrame({"date": dates, "fear_greed": 50})

def calc_indicators(df):
    df = df.copy()
    df["ema_7"]  = df["close"].ewm(span=7,  adjust=False).mean()
    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
    delta = df["close"].diff()
    ag = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + ag / (al + 1e-9))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    bb_u  = sma20 + 2 * std20
    bb_l  = sma20 - 2 * std20
    df["bb_width"]    = (bb_u - bb_l) / (sma20 + 1e-9)
    df["bb_position"] = (df["close"] - bb_l) / (bb_u - bb_l + 1e-9)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"]  - df["close"].shift(1)).abs()
    df["atr_14"] = (pd.concat([hl, hc, lc], axis=1).max(axis=1)
                    .ewm(alpha=1/14, min_periods=14, adjust=False).mean())
    return df

def add_features_1h(df):
    df = df.copy()
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = calc_indicators(df)
    df["return_1"]     = df["close"].pct_change(1)
    df["return_7"]     = df["close"].pct_change(7)
    df["volume_ratio"] = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9)
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.dayofweek / 7)
    return df

def merge_external(df_1h, fg_df):
    df = df_1h.copy()
    df["date"] = df["timestamp"].dt.normalize()
    df = df.merge(fg_df[["date", "fear_greed"]], on="date", how="left")
    df["fear_greed"] = df["fear_greed"].ffill().fillna(50)
    df.drop(columns=["date"], inplace=True)
    return df

def merge_multitf(df_1h, df_4h_raw, df_1d_raw):
    df_4h = calc_indicators(df_4h_raw.copy())
    df_4h["return_4h"] = df_4h["close"].pct_change(1)
    df_4h_f = df_4h[["timestamp", "rsi_14", "macd", "return_4h"]].copy()
    df_4h_f.columns = ["timestamp", "rsi_4h", "macd_4h", "return_4h"]
    
    df_1d = calc_indicators(df_1d_raw.copy())
    df_1d["return_1d"] = df_1d["close"].pct_change(1)
    df_1d_f = df_1d[["timestamp", "rsi_14", "return_1d"]].copy()
    df_1d_f.columns = ["timestamp", "rsi_1d", "return_1d"]
    
    df = pd.merge_asof(df_1h.sort_values("timestamp"),
                       df_4h_f.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    df = pd.merge_asof(df, df_1d_f.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    return df

def add_targets(df, horizons):
    for h in horizons:
        df[f"target_ret_{h}"] = (df["close"].shift(-h) - df["close"]) / df["close"]
    return df

def preprocess_final(df):
    df = df.copy()
    # Outlier clipping para features (no para precios)
    no_price = [c for c in FEATURE_COLS if c not in ["open", "high", "low", "close"]]
    for col in no_price:
        if col in df.columns:
            mu, sigma = df[col].mean(), df[col].std() + 1e-8
            z = (df[col] - mu) / sigma
            df[col] = df[col].where(z.abs() < 10, df[col].median())
    return df.dropna().reset_index(drop=True)

def process_all_preprocessed():
    print("\n-- Preprocesando datasets para RL --------------------------")
    fg_df = download_fear_greed(verbose=True)
    
    for coin in COINS_MAP.keys():
        print(f"  {coin}: procesando...", end=" ")
        p1h = os.path.join(RAW_DIR, f"{coin}_1h_raw.csv")
        p4h = os.path.join(RAW_DIR, f"{coin}_4h_raw.csv")
        p1d = os.path.join(RAW_DIR, f"{coin}_1d_raw.csv")
        
        if not all(os.path.exists(p) for p in [p1h, p4h, p1d]):
            print(f"faltan archivos raw !")
            continue
            
        df_1h = pd.read_csv(p1h, parse_dates=["timestamp"]).sort_values("timestamp")
        df_4h = pd.read_csv(p4h, parse_dates=["timestamp"])
        df_1d = pd.read_csv(p1d, parse_dates=["timestamp"])
        
        for d in [df_1h, df_4h, df_1d]:
            if d["timestamp"].dt.tz is None:
                d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")
                
        df = add_features_1h(df_1h)
        df = merge_external(df, fg_df)
        df = merge_multitf(df, df_4h, df_1d)
        df = add_targets(df, HORIZONS)
        df = preprocess_final(df)
        
        out_path = os.path.join(PREPROCESSED_DIR, f"{coin}_hourly.csv")
        df.to_csv(out_path, index=False)
        print(f"{len(df):,} filas OK")
    
    print("OK data/preprocessed/ actualizado\n")

# -- Main ----------------------------------------------------------------------
if __name__ == "__main__":
    update_local()
    process_all_preprocessed()