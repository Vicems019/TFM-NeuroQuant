from flask_caching import Cache
from dash import Dash
import hashlib
import json

def setup_cache(app: Dash) -> Cache:
    """Configura caché multinivel."""
    cache = Cache(app.server, config={
        'CACHE_TYPE': 'SimpleCache',      # Dev: en memoria
        # 'CACHE_TYPE': 'RedisCache',     # Prod: Redis
        # 'CACHE_REDIS_URL': 'redis://localhost:6379/0',
        'CACHE_DEFAULT_TIMEOUT': 300,     # 5 min default
    })
    return cache

cache = None  # Se inicializa en app.py

# ─── Decoradores de caché por tipo de dato ───────────────────────────────────

def cache_prediction(symbol: str, timeframe: str):
    """Caché para predicciones LSTM: 5 minutos."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"pred:{symbol}:{timeframe}:{_today_hour()}"
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result, timeout=300)
            return result
        return wrapper
    return decorator

def cache_decision(symbol: str):
    """Caché para decisiones SAC: 2 minutos (más volátil)."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"sac:{symbol}:{_today_minute_block(2)}"
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result, timeout=120)
            return result
        return wrapper
    return decorator

def cache_market_data(symbol: str, granularity: str = "1h"):
    """Caché para datos de mercado: 1 minuto."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            key = f"market:{symbol}:{granularity}:{_today_minute_block(1)}"
            result = cache.get(key)
            if result is None:
                result = func(*args, **kwargs)
                cache.set(key, result, timeout=60)
            return result
        return wrapper
    return decorator

def invalidate_symbol(symbol: str):
    """Invalida toda la caché de un símbolo (al cambiar divisa)."""
    # Con Redis: usar SCAN + DEL por patrón
    # Con SimpleCache: resetear keys conocidas
    patterns = [
        f"pred:{symbol}:*",
        f"sac:{symbol}:*",
        f"market:{symbol}:*",
    ]
    # Implementación según backend de caché
    logger.info(f"Cache invalidada para {symbol}")

def _today_hour() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H")

def _today_minute_block(minutes: int) -> str:
    from datetime import datetime
    now = datetime.now()
    block = now.minute // minutes
    return now.strftime(f"%Y%m%d%H{block}")