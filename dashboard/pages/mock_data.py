import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Precios base por cripto (aproximados)
BASE_PRICES = {
    "BTC": 78448,
    "ETH": 3120,
    "SOL": 148,
    "XRP": 0.512,
    "AVAX": 28.4
}

# Métricas de rendimiento simuladas por cripto
METRICS = {
    "BTC": {"retorno_rl": 42.1, "retorno_bh": 16.2, "sharpe": 1.87, "max_dd": -8.3},
    "ETH": {"retorno_rl": 38.5, "retorno_bh": 12.8, "sharpe": 1.64, "max_dd": -11.2},
    "SOL": {"retorno_rl": 61.3, "retorno_bh": 34.1, "sharpe": 2.12, "max_dd": -14.7},
    "XRP": {"retorno_rl": 29.7, "retorno_bh": 8.4,  "sharpe": 1.43, "max_dd": -9.1},
    "AVAX": {"retorno_rl": 47.8, "retorno_bh": 21.3, "sharpe": 1.91, "max_dd": -12.4},
}


def get_precio_actual(cripto: str) -> dict:
    base = BASE_PRICES[cripto]
    variacion = random.uniform(-0.015, 0.015)
    precio = base * (1 + variacion)
    cambio_24h = random.uniform(-2.5, 3.5)
    return {"precio": precio, "cambio_24h": cambio_24h}


def get_historico(cripto: str, dias: int = 7) -> pd.DataFrame:
    """Genera serie histórica simulada de precios OHLCV."""
    base = BASE_PRICES[cripto]
    n = dias * 24  # datos horarios
    np.random.seed(hash(cripto) % 2**31)

    rendimientos = np.random.normal(0.0001, 0.008, n)
    precios = [base * 0.88]
    for r in rendimientos:
        precios.append(precios[-1] * (1 + r))

    fechas = [datetime.utcnow() - timedelta(hours=n - i) for i in range(n + 1)]
    df = pd.DataFrame({
        "fecha": fechas,
        "close": precios,
        "open":  [p * random.uniform(0.997, 1.003) for p in precios],
        "high":  [p * random.uniform(1.001, 1.012) for p in precios],
        "low":   [p * random.uniform(0.988, 0.999) for p in precios],
        "volume":[random.uniform(1e6, 5e6) for _ in precios],
    })
    return df


def get_predicciones_lstm(cripto: str) -> dict:
    """Simula predicciones LSTM para las próximas 4 horas."""
    base = BASE_PRICES[cripto]
    variacion_base = random.uniform(-0.015, 0.015)
    precio_actual = base * (1 + variacion_base)

    tendencia = random.uniform(-0.003, 0.002)  # ligera tendencia bajista o alcista
    pred = {}
    for h in range(1, 5):
        ruido = random.uniform(-0.001, 0.001)
        pred[f"{h}h"] = precio_actual * (1 + tendencia * h + ruido)

    pred["precio_actual"] = precio_actual
    pred["cambio_24h"] = random.uniform(-2.5, 3.5)
    return pred


def get_decision_rl(cripto: str, predicciones: dict) -> dict:
    """Simula decisión del agente RL (PPO)."""
    precio_actual = predicciones.get("precio_actual", BASE_PRICES[cripto])
    precio_4h = predicciones.get("4h", precio_actual)
    cambio_esperado = (precio_4h - precio_actual) / precio_actual

    confianza = random.uniform(0.62, 0.94)

    if cambio_esperado > 0.004:
        accion = "COMPRAR"
    elif cambio_esperado < -0.003:
        accion = "VENDER"
    else:
        accion = "HOLD"

    return {"accion": accion, "confianza": confianza, "cambio_esperado": cambio_esperado * 100}


def get_metricas(cripto: str) -> dict:
    return METRICS[cripto]


def get_backtesting_data(cripto: str) -> pd.DataFrame:
    """Genera curva de equity simulada para backtesting."""
    np.random.seed(hash(cripto + "bt") % 2**31)
    n = 365
    fechas = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n)]

    # Estrategia RL
    rl = [100.0]
    for _ in range(n - 1):
        r = np.random.normal(0.0012, 0.018)
        rl.append(rl[-1] * (1 + r))

    # Buy & Hold
    bh = [100.0]
    for _ in range(n - 1):
        r = np.random.normal(0.0004, 0.022)
        bh.append(bh[-1] * (1 + r))

    return pd.DataFrame({"fecha": fechas, "rl": rl, "bh": bh})


def get_walk_forward_data(cripto: str) -> list:
    """Genera resultados walk-forward por ventana."""
    np.random.seed(hash(cripto + "wf") % 2**31)
    ventanas = []
    fecha = datetime(2023, 6, 1)
    for i in range(8):
        ventanas.append({
            "ventana": i + 1,
            "inicio": (fecha + timedelta(days=i * 45)).strftime("%Y-%m-%d"),
            "fin": (fecha + timedelta(days=(i + 1) * 45)).strftime("%Y-%m-%d"),
            "retorno_rl": round(np.random.normal(5.2, 8.1), 2),
            "retorno_bh": round(np.random.normal(1.8, 6.3), 2),
            "sharpe": round(np.random.uniform(0.8, 2.4), 2),
            "operaciones": int(np.random.randint(18, 65)),
            "win_rate": round(np.random.uniform(0.48, 0.68), 2),
        })
    return ventanas
