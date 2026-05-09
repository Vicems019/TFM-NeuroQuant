import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

BASE_PRICES = {"BTC": 78448, "ETH": 3120, "SOL": 148, "XRP": 0.512, "AVAX": 28.4}
METRICS = {
    "BTC":  {
        "retorno_rl": 42.1, "retorno_bh": 16.2, "sharpe": 1.87, "max_dd": -8.3,
        "mae_lstm": 0.023, "rmse_lstm": 0.031, "r2_lstm": 0.87, "accuracy_lstm": 0.72
    },
    "ETH":  {
        "retorno_rl": 38.5, "retorno_bh": 12.8, "sharpe": 1.64, "max_dd": -11.2,
        "mae_lstm": 0.028, "rmse_lstm": 0.037, "r2_lstm": 0.82, "accuracy_lstm": 0.68
    },
    "SOL":  {
        "retorno_rl": 61.3, "retorno_bh": 34.1, "sharpe": 2.12, "max_dd": -14.7,
        "mae_lstm": 0.035, "rmse_lstm": 0.045, "r2_lstm": 0.79, "accuracy_lstm": 0.65
    },
    "XRP":  {
        "retorno_rl": 29.7, "retorno_bh": 8.4,  "sharpe": 1.43, "max_dd": -9.1,
        "mae_lstm": 0.041, "rmse_lstm": 0.052, "r2_lstm": 0.74, "accuracy_lstm": 0.61
    },
    "AVAX": {
        "retorno_rl": 47.8, "retorno_bh": 21.3, "sharpe": 1.91, "max_dd": -12.4,
        "mae_lstm": 0.032, "rmse_lstm": 0.042, "r2_lstm": 0.81, "accuracy_lstm": 0.67
    },
}
INVERSIONES = {"BTC": 10000, "ETH": 5000, "SOL": 3000, "AVAX": 2000, "ALL": 20000}
CRYPTO_COLORS = {"BTC": "#3b82f6", "ETH": "#10b981", "SOL": "#f59e0b", "AVAX": "#8b5cf6"}

try:
    from .lstm_utils import get_predicciones_lstm_real
    USE_REAL_LSTM = True
except ImportError:
    USE_REAL_LSTM = False
METRICS = {
    "BTC":  {"retorno_rl": 42.1, "retorno_bh": 16.2, "sharpe": 1.87, "max_dd": -8.3},
    "ETH":  {"retorno_rl": 38.5, "retorno_bh": 12.8, "sharpe": 1.64, "max_dd": -11.2},
    "SOL":  {"retorno_rl": 61.3, "retorno_bh": 34.1, "sharpe": 2.12, "max_dd": -14.7},
    "XRP":  {"retorno_rl": 29.7, "retorno_bh": 8.4,  "sharpe": 1.43, "max_dd": -9.1},
    "AVAX": {"retorno_rl": 47.8, "retorno_bh": 21.3, "sharpe": 1.91, "max_dd": -12.4},
}
INVERSIONES = {"BTC": 10000, "ETH": 5000, "SOL": 3000, "AVAX": 2000, "ALL": 20000}
CRYPTO_COLORS = {"BTC": "#3b82f6", "ETH": "#10b981", "SOL": "#f59e0b", "AVAX": "#8b5cf6"}

_HIST_CACHE = {}

def get_historico(cripto: str, dias: int = 7) -> pd.DataFrame:
    try:
        from pathlib import Path
        cripto = cripto.upper()
        
        # Intentar recuperar de la cache primero
        if cripto in _HIST_CACHE:
            df = _HIST_CACHE[cripto]
            n_rows = dias * 24
            return df.tail(n_rows).reset_index(drop=True)
            
        data_path = Path(__file__).parent.parent.parent / "data" / "preprocessed" / f"{cripto}_hourly.csv"
        if data_path.exists():
            # Leer el archivo y guardarlo en cache
            df = pd.read_csv(data_path, parse_dates=["timestamp"])
            df = df.rename(columns={"timestamp": "fecha"})
            _HIST_CACHE[cripto] = df
            
            n_rows = dias * 24
            return df.tail(n_rows).reset_index(drop=True)
    except Exception as e:
        print(f"Error cargando historico real para {cripto}: {e}")

    # Fallback to synthetic
    base = BASE_PRICES.get(cripto, 100)
    n = dias * 24
    np.random.seed(hash(cripto) % 2**31)
    rendimientos = np.random.normal(0.0001, 0.008, n)
    precios = [base * 0.88]
    for r in rendimientos:
        precios.append(precios[-1] * (1 + r))
    fechas = [datetime.utcnow() - timedelta(hours=n - i) for i in range(n + 1)]
    df = pd.DataFrame({
        "fecha":  fechas, "close": precios,
        "open":   [p * random.uniform(0.997, 1.003) for p in precios],
        "high":   [p * random.uniform(1.001, 1.012) for p in precios],
        "low":    [p * random.uniform(0.988, 0.999) for p in precios],
        "volume": [random.uniform(1e6, 5e6) for _ in precios],
    })
    return df

def get_predicciones_lstm(cripto: str) -> dict:
    # Intentar obtener predicción real primero
    res_real = get_predicciones_lstm_real(cripto)
    if res_real and "precio_actual" in res_real:
        return res_real
    
    # Fallback to synthetic
    base = BASE_PRICES.get(cripto, 100)
    variacion_base = random.uniform(-0.015, 0.015)
    precio_actual = base * (1 + variacion_base)
    tendencia = random.uniform(-0.003, 0.002)
    pred = {f"{h}h": precio_actual * (1 + tendencia * h + random.uniform(-0.001, 0.001)) for h in range(1, 5)}
    pred["precio_actual"] = precio_actual
    pred["cambio_24h"] = random.uniform(-2.5, 3.5)
    return pred

def get_decision_rl(cripto: str, predicciones: dict) -> dict:
    precio_actual   = predicciones.get("precio_actual", BASE_PRICES.get(cripto, 100))
    precio_4h       = predicciones.get("4h", precio_actual)
    cambio_esperado = (precio_4h - precio_actual) / precio_actual
    confianza = random.uniform(0.62, 0.94)
    accion = "COMPRAR" if cambio_esperado > 0.004 else ("VENDER" if cambio_esperado < -0.003 else "HOLD")
    return {"accion": accion, "confianza": confianza, "cambio_esperado": cambio_esperado * 100}

def get_metricas(cripto: str) -> dict:
    return METRICS.get(cripto, METRICS["BTC"])

def get_rentabilidad_periodica(cripto: str, periodo: str) -> list:
    """Bars adapted to period: 1d=hourly, 7d=daily, 1m=weekly."""
    np.random.seed(hash(cripto + periodo) % 2**31)
    now = datetime.utcnow()
    if periodo == "1d":
        labels  = [(now - timedelta(hours=23 - i)).strftime("%H:%M") for i in range(24)]
        mu, sig = 0.04, 0.55
    elif periodo == "7d":
        labels  = [(now - timedelta(days=6 - i)).strftime("%a %d/%m") for i in range(7)]
        mu, sig = 0.35, 2.2
    elif periodo == "1m":
        labels  = [f"Sem {i + 1}" for i in range(4)]
        mu, sig = 1.4, 5.5
    else:
        labels  = [(now - timedelta(days=(11 - i) * 30)).strftime("%b %y") for i in range(12)]
        mu, sig = 3.5, 10.0
    retornos = [round(np.random.normal(mu, sig), 2) for _ in labels]
    return [{"label": l, "retorno": r} for l, r in zip(labels, retornos)]

def get_rentabilidad_all(periodo: str) -> dict:
    """All cryptos stacked for the given period."""
    criptos = ["BTC", "ETH", "SOL", "AVAX"]
    labels  = [d["label"] for d in get_rentabilidad_periodica("BTC", periodo)]
    data    = {c: [d["retorno"] for d in get_rentabilidad_periodica(c, periodo)] for c in criptos}
    return {"labels": labels, "data": data}

def get_rentabilidad_absoluta(cripto: str, periodo: str) -> dict:
    """Absolute P&L for period and all-time."""
    inv = INVERSIONES.get(cripto, 10000)
    np.random.seed(hash(cripto + periodo + "abs") % 2**31)
    bases = {"1d": 0.08, "7d": 0.6, "1m": 2.5}
    pct   = round(np.random.normal(bases.get(periodo, 2.5), bases.get(periodo, 2.5) * 0.3), 2)
    total_pct = 16.8 if cripto == "ALL" else 18.4
    return {
        "pct": pct, "abs": round(inv * pct / 100, 2),
        "total_pct": total_pct, "total_abs": round(inv * total_pct / 100, 2),
        "inversion": inv,
    }

def get_backtesting_data(cripto: str) -> pd.DataFrame:
    np.random.seed(hash(cripto + "bt") % 2**31)
    n = 365
    fechas = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    rl = [100.0];  [rl.append(rl[-1] * (1 + np.random.normal(0.0012, 0.018))) for _ in range(n - 1)]
    bh = [100.0];  [bh.append(bh[-1] * (1 + np.random.normal(0.0004, 0.022))) for _ in range(n - 1)]
    return pd.DataFrame({"fecha": fechas, "rl": rl, "bh": bh})

def get_walk_forward_data(cripto: str) -> list:
    np.random.seed(hash(cripto + "wf") % 2**31)
    fecha = datetime(2023, 6, 1)
    return [{
        "ventana": i + 1,
        "inicio":  (fecha + timedelta(days=i * 45)).strftime("%Y-%m-%d"),
        "fin":     (fecha + timedelta(days=(i + 1) * 45)).strftime("%Y-%m-%d"),
        "retorno_rl":  round(np.random.normal(5.2, 8.1), 2),
        "retorno_bh":  round(np.random.normal(1.8, 6.3), 2),
        "sharpe":      round(np.random.uniform(0.8, 2.4), 2),
        "operaciones": int(np.random.randint(18, 65)),
        "win_rate":    round(np.random.uniform(0.48, 0.68), 2),
    } for i in range(8)]

def get_historial_operaciones() -> list:
    random.seed(42)
    criptos = ["BTC", "ETH", "SOL", "AVAX"]
    tipos   = ["COMPRAR", "VENDER", "HOLD"]
    ops, fecha = [], datetime.utcnow()
    for i in range(15):
        tipo   = random.choice(tipos)
        cripto = random.choice(criptos)
        precio = BASE_PRICES.get(cripto, 100) * random.uniform(0.95, 1.05)
        ops.append({
            "id": i + 1, "fecha": (fecha - timedelta(hours=i * 4)).strftime("%d/%m %H:%M"),
            "tipo": tipo, "cripto": cripto, "precio": precio,
            "pnl": round(random.uniform(-8.5, 15.0), 2),
        })
    return ops