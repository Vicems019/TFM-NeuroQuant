import os, json, time, requests
import pandas as pd
from datetime import datetime, timezone

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "raw")
KAGGLE_ID = "vicentelorenzomarn/dataraw"
os.makedirs(RAW_DIR, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
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

# ── Binance ──────────────────────────────────────────────────────────────────
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
    """Descarga histórico completo desde start_date (solo primera vez)."""
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
    Siempre descarga solo las últimas 1000 velas de Binance.
    - Si el CSV existe: añade solo las velas nuevas.
    - Si no existe:     crea el CSV con esas 1000 velas.
    """
    batch = binance_get({"symbol": symbol, "interval": interval, "limit": 1000})
    if batch is None:
        print(f"  {coin} {interval}: sin respuesta de Binance ⚠")
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
            print(f"  {coin} {interval}: ya al día ({last_ts.date()}) ✅")
            return

        df_out = (pd.concat([df_existing, df_new], ignore_index=True)
                    .drop_duplicates("timestamp")
                    .sort_values("timestamp")
                    .reset_index(drop=True))
        df_out.to_csv(path, index=False)
        print(f"  {coin} {interval}: +{len(df_new)} velas → {df_out['timestamp'].max().date()} ✅")
    else:
        df_new.to_csv(path, index=False)
        print(f"  {coin} {interval}: creado con {len(df_new)} velas → {df_new['timestamp'].max().date()} ✅")


def update_local():
    print("\n── Actualizando data/raw/ ──────────────────────────────────")
    for coin, symbol in COINS_MAP.items():
        for interval in ["1h", "4h", "1d"]:
            path = os.path.join(RAW_DIR, f"{coin}_{interval}_raw.csv")
            update_csv(path, symbol, interval, coin)   # sin start_date
    print("✅ data/raw/ actualizado\n")

# ── Push a Kaggle ─────────────────────────────────────────────────────────────
def push_to_kaggle():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
    except ImportError:
        print("⚠ kaggle no instalado: pip install kaggle")
        return

    # El kaggle.json debe estar en ~/.kaggle/kaggle.json con tu API key
    api = KaggleApiExtended()
    api.authenticate()

    # dataset-metadata.json necesario solo si no existe ya en RAW_DIR
    metadata_path = os.path.join(RAW_DIR, "dataset-metadata.json")
    if not os.path.exists(metadata_path):
        metadata = {
            "title": "dataraw",
            "id": KAGGLE_ID,
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    print("── Subiendo a Kaggle ───────────────────────────────────────")
    api.dataset_create_version(
        RAW_DIR,
        version_notes=f"Auto-update {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
        quiet=False,
        delete_old_versions=False,
    )
    print(f"✅ Dataset actualizado: {KAGGLE_ID}\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    update_local()
    push_to_kaggle()