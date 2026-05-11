# pages/api_client.py
import requests, json, os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

root_dir = os.getcwd()
_BASE_URL = None
_CACHE    = {}   # cache por coin para no llamar N veces por página

# ── Leer URL ─────────────────────────────────────────────────
def _get_base_url() -> str:
    global _BASE_URL
    if _BASE_URL:
        return _BASE_URL
    
    config_path = os.path.join(root_dir, "ngrok_url.json")
    
    try:
        with open(config_path) as f:
            data = json.load(f)
            # Priorizamos cloud_url que es el estándar actual
            _BASE_URL = (data.get("cloud_url") or data.get("ngrok_url", "")).rstrip("/")
    except Exception as e:
        logger.error(f"❌ Error leyendo ngrok_url.json: {e}")
        _BASE_URL = "http://localhost:8000" # Fallback
        
    return _BASE_URL

def _prepare_X_input(coin: str) -> list:
    FEATURES = [
    'close', 'volume', 'return_1', 'return_7', 'rsi_14', 'macd',
    'macd_signal', 'bb_position', 'bb_width', 'atr_14', 'volume_ratio',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'fear_greed',
    'return_4h', 'rsi_4h', 'return_1d', 'rsi_1d'
    ]

    preproc_path = os.path.join(root_dir, "data", "preprocessed", f"{coin}_hourly.csv")
    
    if os.path.exists(preproc_path):
        df = pd.read_csv(preproc_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    else:
        logger.warning(f"⚠️ No se encontró datos preprocesados para {coin}. Procesando desde raw...")
        path = os.path.join(root_dir, "data", "raw", f"{coin}_1h_raw.csv")
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Generar features on-the-fly
        df['return_1']  = df['close'].pct_change(1)
        df['return_7']  = df['close'].pct_change(7)
        df['return_4h'] = df['close'].pct_change(4)
        df['return_1d'] = df['close'].pct_change(24)

        def rsi(series, period):
            delta = series.diff()
            gain  = delta.clip(lower=0).rolling(period).mean()
            loss  = (-delta.clip(upper=0)).rolling(period).mean()
            return 100 - (100 / (1 + gain / (loss + 1e-8)))

        df['rsi_14'] = rsi(df['close'], 14)
        df['rsi_4h'] = rsi(df['close'], 4)
        df['rsi_1d'] = rsi(df['close'], 24)

        ema12              = df['close'].ewm(span=12).mean()
        ema26              = df['close'].ewm(span=26).mean()
        df['macd']         = ema12 - ema26
        df['macd_signal']  = df['macd'].ewm(span=9).mean()
        df['macd_4h']      = df['close'].ewm(span=4).mean() - df['close'].ewm(span=8).mean()

        roll               = df['close'].rolling(20)
        bb_mid             = roll.mean()
        bb_std             = roll.std()
        df['bb_position']  = (df['close'] - bb_mid) / (bb_std + 1e-8)
        df['bb_width']     = (2 * bb_std) / (bb_mid + 1e-8)

        high_low           = df['high'] - df['low']
        high_close         = (df['high'] - df['close'].shift()).abs()
        low_close          = (df['low']  - df['close'].shift()).abs()
        df['atr_14']       = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

        df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)

        hour               = df['timestamp'].dt.hour
        dow                = df['timestamp'].dt.dayofweek
        df['hour_sin']     = np.sin(2 * np.pi * hour / 24)
        df['hour_cos']     = np.cos(2 * np.pi * hour / 24)
        df['dow_sin']      = np.sin(2 * np.pi * dow / 7)
        df['dow_cos']      = np.cos(2 * np.pi * dow / 7)

        if 'fear_greed' not in df.columns:
            df['fear_greed'] = 50.0

    df = df[FEATURES].dropna().tail(24)
    X  = df.values.astype(np.float32)

    return X.tolist()  # (24, 21)

# ── Llamada principal ──────────────────────────────────────────
def _call_predict(coin: str) -> dict:
    coin = coin.upper()
    if coin in _CACHE:
        return _CACHE[coin]

    X_input = _prepare_X_input(coin)
    if X_input is None:
        return {}

    try:
        url = _get_base_url()
        endpoint = url if url.endswith("/predict") else f"{url}/predict"
        
        logger.info(f"📡 Enviando petición a Cloud ({coin}) -> {endpoint}")
        
        response = requests.post(
            endpoint,
            json={"symbol": coin, "X_input": X_input},
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"❌ Detalle error servidor: {response.json().get('detail', response.text)}")
            return {}
            
        # TODO QUITAR EL RESPONSE, ES SOLO PARA OBSERVAR
        print("Response: ", response.json())
        res_data = response.json()

        # Guardamos en caché
        _CACHE[coin] = res_data
        return res_data
        
    except Exception as e:
        logger.error(f"❌ Error en llamada a la nube para {coin}: {e}")
        return {}

# ── Precarga de Datos ──────────────────────────────────────────
def preload_all_data():
    logger.info("⏳ Iniciando precarga de datos desde ngrok para todas las monedas...")
    import threading
    def _preload():
        for c in ["BTC", "ETH", "SOL", "AVAX"]:
            _call_predict(c)
        logger.info("✅ Precarga de datos completada con éxito.")
    
    # Run in background to avoid blocking app startup
    threading.Thread(target=_preload, daemon=True).start()

# ── API pública ─────────────────────────────────────────────────

def get_predicciones_lstm_real(coin: str) -> dict:
    data = _call_predict(coin)
    lstm = data["lstm"]
    direction_accuracy = data["direction_accuracy"]
    print("\n\n\n")
    print(f" Coin: {coin} ")
    print(lstm)
    print("\n\n\n")
    return {
        "precio_actual": lstm["predicted_price"],
        "1h":  lstm["1h"],
        "2h":  lstm["2h"],
        "3h":  lstm["3h"],
        "4h":  lstm["4h"],
        "min": lstm["min"],
        "max": lstm["max"],
        "direction_accuracy": direction_accuracy,
    }

def get_metrica(coin: str) -> dict:
    data = _call_predict(coin)
    if not data: return {}
    
    # Las métricas están anidadas en 'metrics'
    m = data.get("metrics", {})
    lstm = data.get("lstm", {})
    
    return {
        "rmse_lstm":     m.get("rmse", 0.0),
        "mae_lstm":      m.get("mae", 0.0),
        "mape_lstm":     m.get("mape", 0.0),
        "accuracy_lstm": lstm.get("direction_accuracy", 0.65),
        "r2_lstm":       m.get("r2", 0.0),
        "sharpe":        m.get("sharpe", 0.0),
    }

def get_decision_rl(coin: str, preds: dict = None) -> dict:
    data = _call_predict(coin)
    if not data: return {"accion": "HOLD", "confianza": 0.5}
    
    # Datos de RL anidados en 'rl'
    rl = data.get("rl", {})
    return {
        "accion":    rl.get("decision", "HOLD"),
        "confianza": rl.get("confidence", 0.5),
    }

def get_lstm_shap(coin: str) -> dict:
    data = _call_predict(coin)
    if not data or "shap" not in data:
        return {"features": ["Error"], "values": [0]}
    
    shap = data["shap"]
    return {
        "features": shap.get("feature_names", []),
        "values":   shap.get("values", []),
    }

def get_rl_shap(coin: str) -> dict:
    # Si el servidor no devuelve SHAP específico para RL, usamos el del LSTM
    data = _call_predict(coin)
    if not data: return {"features": [], "values": []}
    
    if "rl_shap" in data:
        return {"features": data["rl_shap"]["feature_names"], "values": data["rl_shap"]["values"]}
    
    return get_lstm_shap(coin)

def get_trained_rl_metrics(coin: str) -> dict:
    data = _call_predict(coin)
    if not data: return {}
    
    m = data.get("metrics", {})
    rl = data.get("rl", {})
    
    return {
        "sharpe":        m.get("sharpe", 0.0),
        "sortino":       m.get("sortino", m.get("sharpe", 0.0) * 1.1),
        "max_dd":        m.get("max_drawdown", "N/A"),
        "win_rate":      f"{rl.get('confidence', 0.5)*100:.1f}%",
        "profit_factor": m.get("profit_factor", "N/A"),
        "trades":        m.get("total_trades", "N/A"),
    }

def invalidar_cache(coin: str = None):
    global _CACHE
    if coin:
        _CACHE.pop(coin.upper(), None)
    else:
        _CACHE = {}
