import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

@dataclass
class PredictionResult:
    symbol: str
    predictions: np.ndarray      # Precios predichos [t+1..t+n]
    confidence: float             # Intervalo de confianza
    decision: str                 # BUY / SELL / HOLD
    action_probs: np.ndarray      # Probabilidades SAC
    metrics: dict                 # RMSE, MAE, Sharpe, etc.
    latency_ms: float

class PredictionPipeline:
    def __init__(self, model_manager, cache):
        self.mm = model_manager
        self.cache = cache
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    def run(self, symbol: str, market_data: np.ndarray, 
            timeframe: str = "1h") -> PredictionResult:
        """
        Ejecuta LSTM + SAC en paralelo cuando es posible.
        LSTM → features → SAC (secuencial donde se necesita)
        """
        import time
        start = time.perf_counter()

        # Cache hit rápido
        cache_key = f"pipeline:{symbol}:{timeframe}:{hash(market_data.tobytes())}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        # 1. LSTM Predicción
        lstm_result = self._run_lstm(symbol, market_data)
        
        # 2. Preparar estado para SAC + calcular métricas (paralelo)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            sac_future = ex.submit(self._run_sac, symbol, market_data, lstm_result)
            metrics_future = ex.submit(self._compute_metrics, market_data, lstm_result)
            
            decision, action_probs = sac_future.result()
            metrics = metrics_future.result()

        latency = (time.perf_counter() - start) * 1000

        result = PredictionResult(
            symbol=symbol,
            predictions=lstm_result,
            confidence=self._compute_confidence(lstm_result),
            decision=decision,
            action_probs=action_probs,
            metrics=metrics,
            latency_ms=latency
        )

        # Guardar en caché
        self.cache.set(cache_key, result, timeout=120)
        return result

    @torch.no_grad()
    def _run_lstm(self, symbol: str, data: np.ndarray) -> np.ndarray:
        model = self.mm.get_model(f"{symbol}_LSTM")
        if model is None:
            raise RuntimeError(f"Modelo LSTM {symbol} no disponible")
        
        tensor = torch.FloatTensor(data).unsqueeze(0).to(self.mm._device)
        output = model(tensor)
        return output.cpu().numpy().squeeze()

    @torch.no_grad()  
    def _run_sac(self, symbol: str, data: np.ndarray, 
                 lstm_pred: np.ndarray) -> Tuple[str, np.ndarray]:
        model = self.mm.get_model(f"{symbol}_SAC")
        if model is None:
            return "HOLD", np.array([0.33, 0.33, 0.34])
        
        # Estado: datos históricos + predicción LSTM
        state = np.concatenate([data[-20:].flatten(), lstm_pred[:5]])
        tensor = torch.FloatTensor(state).unsqueeze(0).to(self.mm._device)
        
        action_probs = model.get_action_probs(tensor).cpu().numpy().squeeze()
        action_idx = np.argmax(action_probs)
        decision = ["SELL", "HOLD", "BUY"][action_idx]
        
        return decision, action_probs

    def _compute_metrics(self, actual: np.ndarray, 
                         predicted: np.ndarray) -> dict:
        """Métricas en un thread separado, sin bloquear UI."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        n = min(len(actual), len(predicted))
        actual_trim = actual[:n]
        pred_trim = predicted[:n]
        
        rmse = np.sqrt(mean_squared_error(actual_trim, pred_trim))
        mae = mean_absolute_error(actual_trim, pred_trim)
        mape = np.mean(np.abs((actual_trim - pred_trim) / (actual_trim + 1e-8))) * 100
        
        # Sharpe ratio simplificado
        returns = np.diff(predicted) / (predicted[:-1] + 1e-8)
        sharpe = (returns.mean() / (returns.std() + 1e-8)) * np.sqrt(252)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae), 
            'mape': float(mape),
            'sharpe': float(sharpe),
            'direction_accuracy': float(self._direction_acc(actual_trim, pred_trim))
        }

    def _direction_acc(self, actual, predicted) -> float:
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        return np.mean(actual_dir == pred_dir)

    def _compute_confidence(self, predictions: np.ndarray) -> float:
        std = np.std(predictions)
        mean = np.mean(np.abs(predictions))
        cv = std / (mean + 1e-8)
        return float(max(0, 1 - cv))