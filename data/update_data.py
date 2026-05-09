import os, time, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# ── Zona horaria del usuario ──────────────────────────────────────────────────
TZ_MADRID = ZoneInfo("Europe/Madrid")

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
RAW_DIR          = os.path.join(BASE_DIR, "raw")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "preprocessed")

os.makedirs(RAW_DIR,          exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
COINS_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT",
    "SOL": "SOLUSDT", "AVAX": "AVAXUSDT",
}
START_DATES = {
    "BTC": "2020-01-01", "ETH": "2020-01-01",
    "SOL": "2020-09-01", "AVAX": "2020-10-01",
}
INTERVALS    = ["1h", "4h", "1d"]
BINANCE_URLS = [
    "https://api.binance.us/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
]
HORIZONS = [1, 2, 3, 4]

# ── FEATURE_COLS (21 features originales) ────────────────────────────────────
FEATURE_COLS = [
    "close", "volume", "return_1", "return_7",
    "rsi_14", "macd", "macd_signal", "bb_position",
    "bb_width", "atr_14", "volume_ratio", "hour_sin",
    "hour_cos", "dow_sin", "dow_cos", "fear_greed",
    "return_4h", "rsi_4h", "return_1d", "rsi_1d", "macd_4h"
]

columnas_a_eliminar = ['coin', 'ema_7', 'ema_14', 'ema_50']  # Mantenemos atr_14

assert len(FEATURE_COLS) == 21, f"Se esperaban 21 features, hay {len(FEATURE_COLS)}"


# ══════════════════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS RAW
# ══════════════════════════════════════════════════════════════════════════════

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

def _now_madrid():
    return datetime.now(TZ_MADRID).strftime("%Y-%m-%d %H:%M %Z")

def download_historical(symbol, interval, start_date):
    start_ms = _date_to_ms(start_date)
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
                "open":   float(c[1]), "high":  float(c[2]),
                "low":    float(c[3]), "close": float(c[4]),
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

def update_csv(path, symbol, interval, coin, start_date):
    if os.path.exists(path):
        batch = binance_get({"symbol": symbol, "interval": interval, "limit": 1000})
        if batch is None:
            print(f"  {coin} {interval}: sin respuesta de Binance !")
            return

        df_new = pd.DataFrame([{
            "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
            "open":   float(c[1]), "high":  float(c[2]),
            "low":    float(c[3]), "close": float(c[4]),
            "volume": float(c[5]), "coin":  coin,
        } for c in batch])

        df_existing = pd.read_csv(path, parse_dates=["timestamp"])
        if df_existing["timestamp"].dt.tz is None:
            df_existing["timestamp"] = df_existing["timestamp"].dt.tz_localize("UTC")

        last_ts = df_existing["timestamp"].max()
        df_new  = df_new[df_new["timestamp"] > last_ts]

        if df_new.empty:
            print(f"  {coin} {interval}: ya al dia ({last_ts.date()}) OK")
            return

        df_out = (pd.concat([df_existing, df_new], ignore_index=True)
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True))

        df_out = _drop_current_candle(df_out, interval)
        df_out.to_csv(path, index=False)
        print(f"  {coin} {interval}: +{len(df_new)} velas -> {df_out['timestamp'].max().date()} OK")
    else:
        print(f"  {coin} {interval}: descargando historico desde {start_date}...", end=" ", flush=True)
        df = download_historical(symbol, interval, start_date)
        if df is None:
            print("sin datos !")
            return
        df["coin"] = coin
        df.to_csv(path, index=False)
        print(f"{len(df):,} velas -> {df['timestamp'].max().date()} OK")

def update_local():
    print(f"\n[v] Actualizando data/raw/  [{_now_madrid()}]")
    for coin, symbol in COINS_MAP.items():
        for interval in INTERVALS:
            path = os.path.join(RAW_DIR, f"{coin}_{interval}_raw.csv")
            update_csv(path, symbol, interval, coin, START_DATES[coin])
    print("OK data/raw/ actualizado\n")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def download_fear_greed(limit=2000, verbose=True):
    try:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json",
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        df = pd.DataFrame([{
            # Convertir a medianoche de Madrid antes de normalizar
            "date": (pd.to_datetime(int(d["timestamp"]), unit="s", utc=True)
                     .tz_convert("Europe/Madrid")
                     .normalize()
                     .tz_localize(None)),          # quitar tz para el merge
            "fear_greed": int(d["value"]),
        } for d in data]).sort_values("date").reset_index(drop=True)
        if verbose:
            print(f"  Fear & Greed: {len(df)} días (hasta {df['date'].max().date()}) OK")
        return df
    except Exception as e:
        if verbose:
            print(f"  (!) Fear & Greed no disponible ({e}) - usando 50")
        # Fechas de fallback también en Madrid
        dates = (pd.date_range("2020-01-01", periods=2500, freq="D", tz="Europe/Madrid")
                   .tz_localize(None))
        return pd.DataFrame({"date": dates, "fear_greed": 50})


def calc_indicators(df):
    df = df.copy()

    # ── EMAs ──────────────────────────────────────────────────────────────
    df["ema_7"]  = df["close"].ewm(span=7,  adjust=False).mean()
    df["ema_14"] = df["close"].ewm(span=14, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()   

    # ── RSI 14 ────────────────────────────────────────────────────────────
    delta = df["close"].diff()
    ag14  = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    al14  = (-delta).clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    df["rsi_14"] = 100 - 100 / (1 + ag14 / (al14 + 1e-9))

    # ── RSI 6 — más reactivo, captura sobrecompra/venta rápida ────────────
    ag6 = delta.clip(lower=0).ewm(alpha=1/6, min_periods=6, adjust=False).mean()
    al6 = (-delta).clip(lower=0).ewm(alpha=1/6, min_periods=6, adjust=False).mean()
    df["rsi_6"]  = 100 - 100 / (1 + ag6  / (al6  + 1e-9))         

    # ── MACD (12-26-9) ────────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]             

    # ── Bollinger Bands ───────────────────────────────────────────────────
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    bb_u  = sma20 + 2 * std20
    bb_l  = sma20 - 2 * std20
    df["bb_width"]    = (bb_u - bb_l) / (sma20 + 1e-9)
    df["bb_position"] = (df["close"] - bb_l) / (bb_u - bb_l + 1e-9)

    # ── ATR 14 ────────────────────────────────────────────────────────────
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

    # ── Posición de precio ────────────────────────────────────────────────
    df["close_vs_ema7"]  = df["close"] / (df["ema_7"]  + 1e-9) - 1  
    df["close_vs_ema14"] = df["close"] / (df["ema_14"] + 1e-9) - 1  
    df["close_vs_ema50"] = df["close"] / (df["ema_50"] + 1e-9) - 1  
    df["return_1"]       = df["close"].pct_change(1)
    df["return_7"]       = df["close"].pct_change(7)                 
    df["return_4"]       = df["close"].pct_change(4)                 
    df["return_24"]      = df["close"].pct_change(24)                

    # ── Momentum: lags de return_1 ────────────────────────────────────────
    for lag in [1, 2, 3, 4]:
        df[f"return_lag_{lag}"] = df["return_1"].shift(lag)          

    # ── Indicadores (los indicadores base ya están en calc_indicators) ────
    df["atr_14_norm"] = df["atr_14"] / (df["close"] + 1e-9)         

    # ── Volumen ───────────────────────────────────────────────────────────
    df["volume_ratio"]    = df["volume"] / (df["volume"].rolling(24).mean() + 1e-9)
    df["volume_ratio_4h"] = df["volume"] / (df["volume"].rolling(4).mean()  + 1e-9)  

    # ── Estructura de mercado ─────────────────────────────────────────────
    h24 = df["high"].rolling(24).max()
    l24 = df["low"].rolling(24).min()
    df["range_pos_24h"]    = (df["close"] - l24)  / (h24 - l24 + 1e-9)   
    df["dist_to_high_24h"] = (h24 - df["close"])  / (df["close"] + 1e-9)  
    df["dist_to_low_24h"]  = (df["close"] - l24)  / (df["close"] + 1e-9)  

    h7d = df["high"].rolling(7 * 24).max()
    l7d = df["low"].rolling(7 * 24).min()
    df["range_pos_7d"]  = (df["close"] - l7d) / (h7d - l7d + 1e-9)       

    ret_series           = df["close"].pct_change()
    df["volatility_24h"] = ret_series.rolling(24).std()                    
    vol_7d               = ret_series.rolling(7 * 24).std()
    df["vol_ratio"]      = df["volatility_24h"] / (vol_7d + 1e-9)         

    # ── Tiempo cíclico en hora de Madrid ─────────────────────────────────

    ts_mad = df["timestamp"].dt.tz_convert("Europe/Madrid")
    df["hour_sin"]  = np.sin(2 * np.pi * ts_mad.dt.hour   / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * ts_mad.dt.hour   / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * ts_mad.dt.dayofweek / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * ts_mad.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts_mad.dt.month  / 12)          
    df["month_cos"] = np.cos(2 * np.pi * ts_mad.dt.month  / 12)          

    return df


def merge_external(df_1h, fg_df):
    df = df_1h.copy()
    # Fecha en Madrid (sin tz para que coincida con fg_df)
    df["date"] = (df["timestamp"]
                  .dt.tz_convert("Europe/Madrid")
                  .dt.normalize()
                  .dt.tz_localize(None))
    df = df.merge(fg_df[["date", "fear_greed"]], on="date", how="left")
    df["fear_greed"] = df["fear_greed"].ffill().fillna(50)
    df.drop(columns=["date"], inplace=True)
    return df


def merge_multitf(df_1h, df_4h_raw, df_1d_raw):
    # ── 4h ───────────────────────────────────────────────────────────────
    df_4h = calc_indicators(df_4h_raw.copy())
    if df_4h["timestamp"].dt.tz is None:
        df_4h["timestamp"] = df_4h["timestamp"].dt.tz_localize("UTC")
    df_4h["return_4h"] = df_4h["close"].pct_change(1)
    df_4h_f = df_4h[["timestamp", "rsi_14", "return_4h", "macd"]].copy()
    df_4h_f.columns = ["timestamp", "rsi_4h", "return_4h", "macd_4h"]
    df_4h_f["timestamp"] += pd.Timedelta(hours=4)   # anti-lookahead

    # ── 1d ───────────────────────────────────────────────────────────────
    df_1d = calc_indicators(df_1d_raw.copy())
    if df_1d["timestamp"].dt.tz is None:
        df_1d["timestamp"] = df_1d["timestamp"].dt.tz_localize("UTC")
    df_1d["return_1d"] = df_1d["close"].pct_change(1)
    df_1d_f = df_1d[["timestamp", "rsi_14", "return_1d"]].copy()
    df_1d_f.columns = ["timestamp", "rsi_1d", "return_1d"]
    df_1d_f["timestamp"] += pd.Timedelta(days=1)    # anti-lookahead

    df = pd.merge_asof(df_1h.sort_values("timestamp"),
                       df_4h_f.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    df = pd.merge_asof(df,
                       df_1d_f.sort_values("timestamp"),
                       on="timestamp", direction="backward")
    return df


def add_targets(df, horizons):
    for h in horizons:
        df[f"target_ret_{h}"] = (df["close"].shift(-h) - df["close"]) / df["close"]
    return df


def preprocess_final(df):
    df = df.copy()
    numeric_feats = [c for c in FEATURE_COLS if c in df.columns]
    for col in numeric_feats:
        mu, sigma = df[col].mean(), df[col].std() + 1e-8
        z = (df[col] - mu) / sigma
        df[col] = df[col].where(z.abs() < 10, df[col].median())
    return df.dropna().reset_index(drop=True)

def _drop_current_candle(df, interval):
    """Elimina la vela solo si está muy incompleta (menos de 45 mins)."""
    now_utc = pd.Timestamp.now(tz="UTC")
    offsets = {"1h": pd.Timedelta(hours=1),
               "4h": pd.Timedelta(hours=4),
               "1d": pd.Timedelta(days=1)}
    offset = offsets.get(interval, pd.Timedelta(hours=1))
    
    # Si faltan menos de 15 minutos para que cierre la vela, la mantenemos
    limit = now_utc - offset + pd.Timedelta(minutes=15)
    return df[df["timestamp"] <= limit].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def process_all_preprocessed(force=False):
    print(f"\n[v] Preprocesando  [{_now_madrid()}]")
    fg_df = download_fear_greed(verbose=True)

    for coin in COINS_MAP:
        out_path = os.path.join(PREPROCESSED_DIR, f"{coin}_hourly.csv")

        # Comprobar si ya está al día
        if not force and os.path.exists(out_path):
            raw_path  = os.path.join(RAW_DIR, f"{coin}_1h_raw.csv")
            raw_last  = pd.read_csv(raw_path, usecols=["timestamp"],
                                    parse_dates=["timestamp"])["timestamp"].max()
            proc_last = pd.read_csv(out_path, usecols=["timestamp"],
                                    parse_dates=["timestamp"])["timestamp"].max()
            if pd.Timestamp(raw_last) <= pd.Timestamp(proc_last):
                print(f"  {coin}: features al día OK (omitido)")
                continue

        print(f"  {coin}: construyendo features...", end=" ", flush=True)

        # ── Carga de raw ──────────────────────────────────────────────────
        p1h = os.path.join(RAW_DIR, f"{coin}_1h_raw.csv")
        p4h = os.path.join(RAW_DIR, f"{coin}_4h_raw.csv")
        p1d = os.path.join(RAW_DIR, f"{coin}_1d_raw.csv")

        df_1h = pd.read_csv(p1h, parse_dates=["timestamp"])
        df_4h = pd.read_csv(p4h, parse_dates=["timestamp"])
        df_1d = pd.read_csv(p1d, parse_dates=["timestamp"])

        for d in [df_1h, df_4h, df_1d]:
            if d["timestamp"].dt.tz is None:
                d["timestamp"] = d["timestamp"].dt.tz_localize("UTC")

        # ── Pipeline de features ──────────────────────────────────────────
        df = add_features_1h(df_1h)
        df = merge_external(df, fg_df)
        df = merge_multitf(df, df_4h, df_1d)
        df = add_targets(df, HORIZONS)
        df = preprocess_final(df)

        print(df.head())
        print(df.info() )
        print(df.describe())

        df.drop(columns=columnas_a_eliminar, inplace=True)

        print(df.info())
        # ── Verificar que todas las features están presentes ──────────────
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"\n  (!) {coin}: features ausentes -> {missing}")
        else:
            n_ok = len(FEATURE_COLS)
            print(f"{len(df):,} filas | {n_ok}/{n_ok} features OK")

        df.to_csv(out_path, index=False)

    print("OK data/preprocessed/ actualizado\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    update_local()
    process_all_preprocessed(force=True)