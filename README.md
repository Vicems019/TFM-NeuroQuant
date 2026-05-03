# TFM-NeuroQuant

Dashboard interactivo con datos simulados. Listo para conectar tus modelos reales.

## Instalación y ejecución rápida

```bash
# 1. Clonar / descomprimir el proyecto
cd crypto_dashboard

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar
python app.py
```

Abre el navegador en **http://127.0.0.1:8050**

---

## Estructura

```
crypto_dashboard/
├── app.py                   # Entry point
├── requirements.txt
├── assets/
│   └── style.css            # Tema oscuro
├── data/
│   └── mock_data.py         # ← SUSTITUIR por tus datos reales
└── pages/
    ├── vision_general.py    # Tab principal
    ├── backtesting.py       # Tab backtesting
    └── walk_forward.py      # Tab walk-forward
```

## Cómo conectar tus modelos reales

### 1. LSTM — predicciones de precio

En `data/mock_data.py`, sustituye `get_predicciones_lstm()` por tu lógica real:

```python
def get_predicciones_lstm(cripto: str) -> dict:
    df = descargar_datos(cripto, timeframe="1h", limit=200)  # ccxt / yfinance
    modelo = cargar_modelo_lstm(cripto)                       # torch.load / keras
    scaler = cargar_scaler(cripto)
    
    features = preprocesar(df)
    predicciones = modelo.predict(features)                   # shape (4,)
    predicciones = scaler.inverse_transform(predicciones)
    
    return {
        "precio_actual": df["close"].iloc[-1],
        "cambio_24h":    calcular_cambio_24h(df),
        "1h": float(predicciones[0]),
        "2h": float(predicciones[1]),
        "3h": float(predicciones[2]),
        "4h": float(predicciones[3]),
    }
```

### 2. RL PPO — decisión de trading

```python
def get_decision_rl(cripto: str, predicciones: dict) -> dict:
    df = descargar_datos(cripto, timeframe="1h", limit=200)
    agente = cargar_agente_ppo(cripto)                        # stable-baselines3
    
    obs = construir_observacion(df, predicciones)
    accion, _states = agente.predict(obs, deterministic=True)
    
    mapa = {0: "HOLD", 1: "COMPRAR", 2: "VENDER"}
    return {
        "accion":    mapa[int(accion)],
        "confianza": float(agente.policy.get_distribution(obs).distribution.probs.max()),
    }
```
