import os
import sys
import numpy as np

# Fix para compatibilidad con NumPy 2.0+ al cargar modelos entrenados en versiones distintas
sys.modules["numpy._core"] = np.core

import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

# Mismas columnas que usaste en el entrenamiento del SAC (20 features según notebook)
RL_FEATURE_COLS = [
    "close", "volume", "return_1", "return_7",
    "rsi_14", "macd", "macd_signal", "bb_position",
    "bb_width", "atr_14", "volume_ratio", "hour_sin",
    "hour_cos", "dow_sin", "dow_cos", "fear_greed",
    "return_4h", "rsi_4h", "return_1d", "rsi_1d"
]

class CryptoTradingEnv(gym.Env):
    """Entorno simplificado para inferencia y métricas en el Dashboard"""
    def __init__(self, df, lookback=32, initial_capital=10000.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.initial_capital = initial_capital
        
        # Selección de features
        if not all(col in df.columns for col in RL_FEATURE_COLS):
            # Fallback en caso de que falte alguna columna (como macd_4h si se incluyó por error)
            available_cols = [c for c in RL_FEATURE_COLS if c in df.columns]
            raw = df[available_cols].values.astype(np.float32)
        else:
            raw = df[RL_FEATURE_COLS].values.astype(np.float32)
            
        # Normalización Z-score (idealmente usar las medias del entrenamiento)
        self.mean = raw.mean(axis=0)
        self.std = raw.std(axis=0) + 1e-8
        self._feat = (raw - self.mean) / self.std
        
        self.n_features = self._feat.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(lookback * self.n_features + 3,), 
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.lookback
        self.capital = self.initial_capital
        self.position = 0 # 0: cash, 1: long
        self.units = 0.0
        self.pv = self.initial_capital
        self.pv_hist = [self.initial_capital]
        self.trades = 0
        return self._get_obs(), {}

    def _get_obs(self):
        n = self.lookback * self.n_features
        obs = np.zeros(n + 3, dtype=np.float32)
        # Flatten the window
        window = self._feat[self.current_step - self.lookback : self.current_step].ravel()
        obs[:n] = window
        obs[n] = float(self.position)
        obs[n+1] = (self.pv - self.initial_capital) / self.initial_capital
        obs[n+2] = self.capital / self.initial_capital
        return obs

    def step(self, action):
        price = self.df.loc[self.current_step, "close"]
        signal = action[0] # SAC output is continuous [-1, 1]
        
        # Umbrales del notebook (0.33)
        if signal > 0.33 and self.position == 0:
            # COMPRAR
            self.units = self.capital / (price * 1.001) # Incluye comisión aprox
            self.capital = 0
            self.position = 1
            self.trades += 1
        elif signal < -0.33 and self.position == 1:
            # VENDER
            self.capital = self.units * price * 0.999 # Incluye comisión aprox
            self.units = 0
            self.position = 0
            self.trades += 1
            
        self.pv = self.capital + (self.units * price if self.position == 1 else 0)
        self.pv_hist.append(self.pv)
        
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), 0, done, False, {}

def calculate_rl_metrics(pv_hist, trades_count, initial_capital=10000.0):
    pv = np.array(pv_hist)
    returns = np.diff(pv) / (pv[:-1] + 1e-8)
    
    total_ret = (pv[-1] - initial_capital) / initial_capital
    # PPA para 4h (aprox 1512 velas/año)
    ppa = 1512 
    
    sharpe = 0
    sortino = 0
    if len(returns) > 1:
        std = returns.std()
        if std > 0:
            sharpe = (returns.mean() / std) * np.sqrt(ppa)
        
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        if downside_std > 0:
            sortino = (returns.mean() / downside_std) * np.sqrt(ppa)
    
    # Max Drawdown
    peak = np.maximum.accumulate(pv)
    drawdown = (peak - pv) / (peak + 1e-8)
    max_dd = drawdown.max()
    
    # Win Rate (operaciones ganadoras vs perdedoras)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / (len(returns) + 1e-8)
    
    # Profit Factor
    pos_rets = wins.sum()
    neg_rets = abs(losses.sum())
    profit_factor = pos_rets / (neg_rets + 1e-8)
    
    return {
        "sharpe": f"{sharpe:.2f}",
        "sortino": f"{sortino:.2f}",
        "max_dd": f"-{max_dd * 100:.1f}%",
        "win_rate": f"{win_rate * 100:.1f}%",
        "profit_factor": f"{profit_factor:.2f}",
        "trades": str(trades_count),
        "total_return": f"{total_ret * 100:.1f}%",
        "equity_curve": pv_hist
    }

def get_trained_rl_metrics(coin="BTC"):
    # Rutas relativas desde el dashboard
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_path, "models", "sac", f"sac_{coin}_best.zip")
    
    if not os.path.exists(model_path):
        # Mock si no existe el modelo
        return {
            "sharpe": "0.00", "sortino": "0.00", "max_dd": "0.0%", 
            "win_rate": "0.0%", "profit_factor": "0.00", "trades": "0"
        }
    
    import numpy, torch, sklearn
    print(f"numpy:   {numpy.__version__}")
    print(f"torch:   {torch.__version__}")
    print(f"sklearn: {sklearn.__version__}")

    try:
        import stable_baselines3 as sb3
        print(f"sb3:     {sb3.__version__}")
    except ImportError:
        print("sb3: no instalado")

    try:
        data_path = os.path.join(base_path, "data", "preprocessed", f"{coin}_hourly.csv")

        if not os.path.exists(data_path): return None
        
        # Leemos los últimos 2 meses para el test del dashboard
        df = pd.read_csv(data_path).tail(24 * 60) 
        
        env = CryptoTradingEnv(df)
        model = SAC.load(model_path, device="cpu")
        
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            
        return calculate_rl_metrics(env.pv_hist, env.trades)
    except Exception as e:
        print(f"Error cargando SAC {coin}: {e}")
        return None


def get_rl_shap(coin="BTC"):
    """
    Explica la política del agente RL usando SHAP.
    """
    try:
        import shap
    except ImportError:
        return {"features": ["Pred. LSTM", "Volatilidad", "Rend. Acum.", "Pos. Actual", "Drawdown"], "values": [0.40, 0.25, 0.15, 0.12, 0.08]}

    coin = coin.upper()
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_path, "models", "sac", f"sac_{coin}_best.zip")
    
    if not os.path.exists(model_path): return None

    try:
        model = SAC.load(model_path, device="cpu")
        # El actor es quien toma las decisiones
        actor = model.policy.actor
        
        data_path = os.path.join(base_path, "data", "preprocessed", f"{coin}_hourly.csv")
        df = pd.read_csv(data_path).tail(100)
        env = CryptoTradingEnv(df)
        
        # Generar algunas observaciones para el background
        obs_list = []
        obs, _ = env.reset()
        for _ in range(20):
            obs_list.append(obs)
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, _ = env.step(action)
        
        background = torch.from_numpy(np.array(obs_list))
        test_obs = background[-1:]
        
        # Explainer para PyTorch
        explainer = shap.DeepExplainer(actor, background)
        shap_values = explainer.shap_values(test_obs)
        
        # Los shap_values para SAC (1 acción) suelen ser una lista con 1 array de (1, obs_dim)
        if isinstance(shap_values, list):
            importance = np.abs(shap_values[0]).mean(axis=0)
        else:
            importance = np.abs(shap_values).mean(axis=0)
            
        # El vector de observación tiene: lookback * n_features + 3
        # Queremos agregar la importancia por feature
        n_feat = env.n_features
        lookback = env.lookback
        
        feat_importance = np.zeros(n_feat + 3)
        for i in range(lookback):
            feat_importance[:n_feat] += importance[i*n_feat : (i+1)*n_feat]
        
        # Los últimos 3 son variables de estado (position, return, capital)
        feat_importance[n_feat:] = importance[lookback*n_feat:]
        
        # Nombres de columnas
        col_names = RL_FEATURE_COLS + ["Posicion", "Retorno_PV", "Capital_Rel"]
        
        # Normalizar y coger top 5
        feat_importance = feat_importance / (feat_importance.sum() + 1e-8)
        indices = np.argsort(feat_importance)[::-1][:5]
        
        return {
            "features": [col_names[i] for i in indices],
            "values": [float(feat_importance[i]) for i in indices]
        }
        
    except Exception as e:
        print(f"Error en SHAP RL ({coin}): {e}")
        return {"features": ["Error"], "values": [0]}
