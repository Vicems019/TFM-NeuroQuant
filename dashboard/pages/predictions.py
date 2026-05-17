import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import sys, os
import pandas as pd
import numpy as np
from datetime import timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.api_client import (
    get_metrica, get_lstm_shap, get_predicciones_lstm_real,
    get_trained_rl_metrics, get_rl_shap, get_decision_rl
)
from pages.currency_utils import format_price, CURRENCY_RATES


dash.register_page(__name__, path="/predictions", name="Predicciones IA")


def _fmt_window_diracc(v):
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)[:14]
    if -1 <= fv <= 1:
        return f"{fv * 100:.2f}%"
    return f"{fv:.4f}"


def _fmt_api_float(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)[:20]


def _fmt_rl_confidence(c):
    if c is None:
        return "—"
    try:
        fc = float(c)
    except (TypeError, ValueError):
        return str(c)[:12]
    if 0 <= fc <= 1:
        return f"{fc * 100:.1f}%"
    return f"{fc:.4f}"


def _fmt_signal_agreement(sa):
    if sa is None:
        return "—"
    if isinstance(sa, bool):
        return "Sí" if sa else "No"
    if isinstance(sa, (int, float)):
        return f"{float(sa):.4f}"
    return str(sa)[:24]


# Histórico visible en el gráfico (1 semana de velas 1h)
CHART_HISTORY_HOURS = 24 * 7


def _sanitize_forecast_band(p0, pred_vals, raw_min, raw_max):
    """
    Banda [min,max] del API alineada con la trayectoria LSTM: en cada hora i la predicción
    queda dentro del cono; en t=+4h los bordes coinciden con min/max del API (tras saneo).
    """
    try:
        p0 = float(p0)
        pv = [float(x) for x in pred_vals]
    except (TypeError, ValueError):
        return None, None, [], []
    if len(pv) != 4:
        return None, None, [], []
    path_lo = min(p0, min(pv))
    path_hi = max(p0, max(pv))
    p4 = pv[3]
    eps = max(abs(p0) * 5e-4, 1e-8)

    if raw_min is None or raw_max is None:
        span = max(path_hi - path_lo, eps * 10)
        fmin = path_lo - span * 0.15
        fmax = path_hi + span * 0.15
    else:
        try:
            lo, hi = float(raw_min), float(raw_max)
        except (TypeError, ValueError):
            span = max(path_hi - path_lo, eps * 10)
            fmin = path_lo - span * 0.15
            fmax = path_hi + span * 0.15
        else:
            if lo > hi:
                lo, hi = hi, lo
            if hi - lo < eps:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - eps, mid + eps
            fmin, fmax = lo, hi

    fmin = min(fmin, path_lo, p4) - eps
    fmax = max(fmax, path_hi, p4) + eps
    pad = max((fmax - fmin) * 0.015, eps)
    fmin -= pad
    fmax += pad

    mup = max(fmax - p4, eps)
    mdn = max(p4 - fmin, eps)
    upper, lower = [], []
    for i in range(1, 5):
        pi = pv[i - 1]
        frac = i / 4.0
        upper.append(pi + frac * mup)
        lower.append(pi - frac * mdn)
    return fmin, fmax, upper, lower


def load_historical_data(coin="BTC"):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", f"{coin}_1h_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            # Convertir a horario de Madrid para visualización
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize("UTC")
            df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Madrid")
            df = df.sort_values("timestamp")
            return df.tail(24 * 30)
    return pd.DataFrame()

# ── MODAL INFO ──────────────────────────────────────────────
modal_info = html.Div([
    html.Div(id="modal-backdrop-info", n_clicks=0, style={"position": "absolute", "inset": 0, "background": "rgba(4,8,15,0.85)", "backdropFilter": "blur(4px)"}),
    html.Div([
        html.Button("✕", id="modal-close-x-info", n_clicks=0, style={"position": "absolute", "top": "15px", "right": "20px", "background": "transparent", "border": "none", "color": "#94a3b8", "fontSize": "22px", "cursor": "pointer", "zIndex": "10"}),
        html.Div([
            html.Span("ℹ¿Qué es Bitcoin?", className="modal-title-text"),
        ], className="modal-header-row", style={"borderBottom": "1px solid rgba(59,130,246,0.13)", "paddingBottom": "15px"}),
        html.Div([
            html.P(
                "Bitcoin (BTC) es una red de pagos descentralizada y una criptomoneda creada en 2009 por un ente seudónimo llamado Satoshi Nakamoto. "
                "Funciona sin una autoridad central o administrador único, lo que significa que el dinero puede enviarse de un usuario a otro en la red peer-to-peer de Bitcoin sin necesidad de intermediarios.",
                style={"color": "#94a3b8", "lineHeight": "1.6", "fontSize": "15px"}
            ),
            html.P(
                "A menudo se describe como 'oro digital' debido a su suministro limitado (21 millones de monedas como máximo) y su resistencia a la censura. "
                "Las transacciones se verifican mediante criptografía y se registran en un libro de contabilidad público distribuido llamado blockchain.",
                style={"color": "#94a3b8", "lineHeight": "1.6", "fontSize": "15px", "marginTop": "15px"}
            ),
        ], className="modal-body-custom", style={"padding": "15px 0 0 0"}),
    ], style={"width": "600px", "maxWidth": "95vw", "background": "#080e1e", "padding": "32px", "borderRadius": "16px", "position": "relative", "boxShadow": "0 10px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.13)", "zIndex": "1001"}),
], id="modal-wrapper-info", style={"display": "none"})

# ── MODAL METRICAS Y EXPLICABILIDAD ─────────────────────────
modal_metrics = html.Div([
    html.Div(id="modal-backdrop-metrics", n_clicks=0, style={"position": "absolute", "inset": 0, "background": "rgba(4,8,15,0.85)", "backdropFilter": "blur(4px)"}),
    html.Div([
        html.Button("✕", id="modal-close-x-metrics", n_clicks=0, style={"position": "absolute", "top": "15px", "right": "20px", "background": "transparent", "border": "none", "color": "#94a3b8", "fontSize": "22px", "cursor": "pointer", "zIndex": "10"}),
        html.Div([
            html.Span("Métricas y Explicabilidad", className="modal-title-text"),
        ], className="modal-header-row", style={"borderBottom": "1px solid rgba(59,130,246,0.13)", "paddingBottom": "15px"}),
        
        html.Div([
            # LSTM Metrics
            html.Div([
                html.H4("Modelo Predictivo (LSTM)", style={"color": "white", "marginBottom": "15px"}),
                html.Div([
                    html.Div([html.Span("RMSE", style={"fontWeight": "bold"}), html.Div(id="rmse-value", className="pnl-val"), html.Span("Error cuadrático medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("MAE", style={"fontWeight": "bold"}), html.Div(id="mae-value", className="pnl-val"), html.Span("Error absoluto medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("MAPE", style={"fontWeight": "bold"}), html.Div(id="mape-value", className="pnl-val"), html.Span("Error porcentual medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Dir. acc. (ventana)", style={"fontWeight": "bold"}), html.Div(id="dir-acc-value", className="pnl-val green"), html.Span("Acierto de dirección en ventana reciente", className="modal-hint")], className="pnl-stat-card"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"})
            ]),
            
            # RL Metrics
            html.Div([
                html.H4("Modelo de Decisión (RL SAC)", style={"color": "white", "margin": "20px 0 15px 0"}),
                html.Div([
                    html.Div([html.Span("Sharpe ratio", style={"fontWeight": "bold"}), html.Div(id="sharpe-rl", className="pnl-val green"), html.Span("Retorno ajustado por riesgo", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Retorno total", style={"fontWeight": "bold"}), html.Div(id="total-return-rl", className="pnl-val green"), html.Span("Acumulado en backtest", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Calmar", style={"fontWeight": "bold"}), html.Div(id="calmar-rl", className="pnl-val green"), html.Span("Rentabilidad vs drawdown máximo", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Confianza RL", style={"fontWeight": "bold"}), html.Div(id="rl-confidence", className="pnl-val green"), html.Span("Probabilidad asignada a la acción", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Acuerdo señales", style={"fontWeight": "bold"}), html.Div(id="signal-agreement-rl", className="pnl-val green"), html.Span("Coherencia entre señales", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Acción raw", style={"fontWeight": "bold"}), html.Div(id="raw-action-rl", className="pnl-val"), html.Span("Salida continua del agente", className="modal-hint")], className="pnl-stat-card"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "10px"}),
            ]),

            # Explicabilidad with Scroll & Bar Charts
            html.Div([
                html.H4("Explicabilidad (Importancia de Atributos)", style={"color": "white", "margin": "20px 0 15px 0"}),
                html.Div([
                    html.Div([
                        html.Div("Factores predictivos (LSTM)", style={"fontWeight": "bold", "color": "#3b82f6", "marginBottom": "5px"}),
                        dcc.Graph(id="chart-expl-lstm", config={"displayModeBar": False}, style={"height": "220px"})
                    ], className="pnl-stat-card", style={"flex": "1", "minWidth": "350px"}),
                    html.Div([
                        html.Div("Pesos del Estado (RL SAC)", style={"fontWeight": "bold", "color": "#10b981", "marginBottom": "5px"}),
                        dcc.Graph(id="chart-expl-sac", config={"displayModeBar": False}, style={"height": "220px"})
                    ], className="pnl-stat-card", style={"flex": "1", "minWidth": "350px"}),
                ], style={"display": "flex", "gap": "15px", "overflowX": "auto", "paddingBottom": "10px"})
            ])
            
        ], className="modal-body-custom", style={"padding": "15px 0 0 0", "maxHeight": "70vh", "overflowY": "auto", "overflowX": "hidden"}),
    ], style={"width": "1100px", "maxWidth": "98vw", "background": "#080e1e", "padding": "32px", "borderRadius": "16px", "position": "relative", "boxShadow": "0 10px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(59,130,246,0.13)", "zIndex": "1001"}),
], id="modal-wrapper-metrics", style={"display": "none"})

# ── LAYOUT ──────────────────────────────────────────────────
layout = html.Div([
    modal_info,
    modal_metrics,
    
    # Header: Title + Info Button
    html.Div([
        html.Span("Predicciones IA", id="page-title-btc", className="section-title", style={"fontSize": "1.5rem"}),
        html.Button("ℹ️ Info", id="btn-info-btc", className="btn-add-op", n_clicks=0, style={"width": "auto", "padding": "0 15px"})
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}),
    
    # Top Row: 2 Horizontal Cards
    html.Div([
        # Card 1: Current Price & 24h Variation
        html.Div([
            html.Div([
                html.Div([
                    html.Div("Precio Actual", className="pnl-label"),
                    html.Div(id="card-current-price", className="pnl-val", style={"fontSize": "2.2rem", "fontWeight": "800"}),
                ]),
                html.Div(id="card-price-change", style={"textAlign": "right", "fontWeight": "bold"})
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start"}),
            
            # La Franja (Range Bar)
            html.Div(id="franja-container", style={"marginTop": "25px"})
        ], className="pnl-stat-card", style={"flex": "1", "position": "relative", "padding": "25px"}),
        
        # Card 2: Clickable Metrics + Explainability
        html.Div([
            html.Div([
                html.Span("Métricas y Explicabilidad", className="pnl-label", style={"fontSize": "1.1rem"}),
                html.Span("➔", style={"color": "#3b82f6", "fontSize": "1.5rem"}),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
            html.P("Explora la interpretación de los modelos, importancia de características y métricas de error.", 
                   style={"color": "#94a3b8", "marginTop": "15px", "lineHeight": "1.5"})
        ], id="btn-open-metrics", className="pnl-stat-card", style={"flex": "1", "cursor": "pointer", "padding": "25px", "transition": "transform 0.2s"})
    ], style={"display": "flex", "gap": "20px", "marginBottom": "25px"}),
    
    # Main Row: Left Chart, Right Predictions
    html.Div([
        # Left: Chart with Uncertainty Cone
        html.Div([
            html.Div([
                html.Span("Proyección de Precios a 4 Horas", className="section-title"),
                html.Span(" · últimos 7 días + 4 h (tiempo real)", style={"color": "#64748b", "fontSize": "12px", "fontWeight": "500", "marginLeft": "8px"}),
            ], style={"marginBottom": "15px"}),
            dcc.Graph(id="grafico-prediccion-cono", config={"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToAdd": ['zoomIn2d', 'zoomOut2d']}, style={"height": "530px"})
        ], className="panel-bar-chart", style={"flex": "2.2", "padding": "25px", "display": "flex", "flexDirection": "column"}),
        
        # Right: Predictions Column
        html.Div([
            html.Div([
                html.Span("Predicciones (LSTM)", className="section-title"),
                html.Div(id="lista-predicciones-4h", style={"marginTop": "20px", "display": "flex", "flexDirection": "column", "gap": "15px"}),
            ]),
            
            html.Div([
                html.Span("Recomendación (RL SAC)", className="section-title"),
                html.Div(id="caja-decision-rl", style={"marginTop": "20px", "display": "flex", "alignItems": "center", "justifyContent": "center", "flex": "1"})
            ], style={"display": "flex", "flexDirection": "column", "flex": "1", "marginTop": "45px"})
            
        ], className="panel-ops-table", style={"flex": "1", "display": "flex", "flexDirection": "column", "justifyContent": "space-between", "padding": "25px", "minHeight": "600px"})
    ], style={"display": "flex", "gap": "20px", "alignItems": "stretch"})
    
], className="page-container")

# ── CALLBACKS ───────────────────────────────────────────────

@callback(
    Output("modal-wrapper-info", "style"),
    Input("btn-info-btc", "n_clicks"),
    Input("modal-close-x-info", "n_clicks"),
    Input("modal-backdrop-info", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal_info(btn, close_x, backdrop):
    ctx = dash.callback_context
    if not ctx.triggered:
        return {"display": "none"}
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "btn-info-btc":
        return {"display": "flex", "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%", "zIndex": "1000", "alignItems": "center", "justifyContent": "center"}
    return {"display": "none"}

@callback(
    Output("modal-wrapper-metrics", "style"),
    Output("chart-expl-lstm", "figure"),
    Output("chart-expl-sac", "figure"),
    Input("btn-open-metrics", "n_clicks"),
    Input("modal-close-x-metrics", "n_clicks"),
    Input("modal-backdrop-metrics", "n_clicks"),
    State("store-cripto", "data"),
    prevent_initial_call=True,
)
def toggle_modal_metrics(btn, close_x, backdrop, cripto):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "btn-open-metrics":
        if not cripto: cripto = "BTC"

        # ── Obtener datos ──────────────────────────────────────────────
        expl_lstm_raw = get_lstm_shap(cripto)
        expl_rl_raw   = get_rl_shap(cripto)
        top_lstm      = expl_lstm_raw.get("top_features", [])
        top_rl        = expl_rl_raw.get("top_features",  [])

        # ── Chart RL ilustrativo (sin cambios) ────────────────────────
        rl_metrics = get_trained_rl_metrics(cripto) or {}
        total_ret  = float(rl_metrics.get("total_return") or 16.0)
        bh_ret     = float(rl_metrics.get("bh_return")    or -24.0)
        sharpe     = float(rl_metrics.get("sharpe")        or 0.5)

        # ── Chart LSTM SHAP ───────────────────────────────────────────
        lstm_names  = [f["feature"]    for f in top_lstm]
        lstm_vals   = [f["shap_value"] for f in top_lstm]
        lstm_colors = ["#10b981" if v >= 0 else "#ef4444" for v in lstm_vals]
        lstm_hover  = [
            f"<b>{f['feature']}</b><br>SHAP: {f['shap_value']:+.6f}<br>Efecto: {f['direction']}"
            for f in top_lstm
        ]

        fig_lstm = go.Figure(go.Bar(
            x=lstm_vals,
            y=lstm_names,
            orientation="h",
            marker=dict(color=lstm_colors, line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
            text=[f"{v:+.5f}" for v in lstm_vals],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=lstm_hover,
        ))
        fig_lstm.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        fig_lstm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=60, t=20, b=5),
            title=dict(text="← Baja precio  /  Sube precio →", font=dict(color="#64748b", size=9), x=0.5),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#e2e8f0", size=10),
        )

        # ── Chart SAC SHAP + portfolio sensitivity ────────────────────
        rl_names  = [f["feature"]    for f in top_rl]
        rl_vals   = [f["shap_value"] for f in top_rl]
        rl_colors = ["#10b981" if v >= 0 else "#ef4444" for v in rl_vals]
        rl_hover  = [
            f"<b>{f['feature']}</b><br>SHAP: {f['shap_value']:+.6f}<br>Efecto: {f['direction']}"
            for f in top_rl
        ]

        ps = expl_rl_raw.get("portfolio_sensitivity", {})
        portfolio_extra = [
            ("position (long)",   ps.get("long_position_impact",   0)),
            ("position (short)",  ps.get("short_position_impact",  0)),
            ("pnl_impact",        ps.get("pnl_impact_5pct",        0)),
            ("capital",           ps.get("reduced_capital_impact", 0)),
        ]
        for label, val in portfolio_extra:
            if val != 0:
                rl_names.append(label)
                rl_vals.append(val)
                rl_colors.append("#3b82f6" if val >= 0 else "#f59e0b")
                rl_hover.append(f"<b>{label}</b><br>Δacción: {val:+.6f}<br>Sensitivity analysis")

        fig_sac = go.Figure(go.Bar(
            x=rl_vals,
            y=rl_names,
            orientation="h",
            marker=dict(color=rl_colors, line=dict(color="rgba(255,255,255,0.08)", width=0.5)),
            text=[f"{v:+.5f}" for v in rl_vals],
            textposition="outside",
            textfont=dict(color="#e2e8f0", size=9),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=rl_hover,
        ))
        fig_sac.add_vline(x=0, line_width=1, line_color="rgba(255,255,255,0.2)")
        fig_sac.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=60, t=20, b=5),
            title=dict(text="← Pro-venta  /  Pro-compra →", font=dict(color="#64748b", size=9), x=0.5),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, showticklabels=False),
            yaxis=dict(autorange="reversed"),
            font=dict(color="#e2e8f0", size=10),
        )

        return (
            {"display": "flex", "position": "fixed", "top": "0", "left": "0",
            "width": "100%", "height": "100%", "zIndex": "1000",
            "alignItems": "center", "justifyContent": "center"},
            fig_lstm, fig_sac
        )

    return {"display": "none"}, dash.no_update, dash.no_update

def _build_equity_curve(total_return: float, sharpe: float, n: int = 180) -> np.ndarray:
    vol      = max(0.05, 0.12 / max(abs(sharpe), 0.1))
    drift    = (total_return / 100) / n
    rng      = np.random.default_rng(seed=42)
    steps    = rng.normal(drift, vol / np.sqrt(252), n)
    curve    = np.cumsum(steps) * 100
    if curve[-1] != 0:
        curve = curve * (total_return / curve[-1])
    return curve



@callback(
    Output("page-title-btc", "children"),
    Output("card-current-price", "children"),
    Output("card-price-change", "children"),
    Output("card-price-change", "style"),
    Output("grafico-prediccion-cono", "figure"),
    Output("lista-predicciones-4h", "children"),
    Output("caja-decision-rl", "children"),
    Output("rmse-value", "children"),
    Output("mae-value", "children"),
    Output("mape-value", "children"),
    Output("dir-acc-value", "children"),
    Output("sharpe-rl", "children"),
    Output("total-return-rl", "children"),
    Output("calmar-rl", "children"),
    Output("rl-confidence", "children"),
    Output("signal-agreement-rl", "children"),
    Output("raw-action-rl", "children"),
    Output("franja-container", "children"),
    Input("store-cripto", "data"),
    Input("store-currency", "data"),
)
def update_predictions_page(cripto, currency):
    if not currency: currency = "USD"
    if not cripto or cripto == "ALL":
        cripto = "BTC"
        
    names = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "AVAX": "Avalanche", "XRP": "Ripple"}
    name = names.get(cripto, cripto)
    title_str = f"{name} ({cripto}) - Predicciones IA"
    
    # 1. Get Real Predictions
    try:
        preds = get_predicciones_lstm_real(cripto)
    except Exception as e:
        print(f"Error en inferencia real para {cripto}: {e}")
        preds = {}
        
    api_current_price = preds.get("precio_actual", 0)

    # 2. Load Data for chart
    df = load_historical_data(cripto)
    if not df.empty and len(df) >= 25:
        current_price = api_current_price if api_current_price > 0 else df.iloc[-1]['close']
        last_time = df.iloc[-1]['timestamp']
        if len(df) >= CHART_HISTORY_HOURS + 1:
            prev_ref_price = float(df.iloc[-(CHART_HISTORY_HOURS + 1)]["close"])
            change_ref_lbl = "vs hace 7 días"
        else:
            prev_ref_price = float(df.iloc[-25]["close"])
            change_ref_lbl = "vs ~24 h"
    else:
        current_price = api_current_price if api_current_price > 0 else 100
        prev_ref_price = current_price * 0.98
        last_time = pd.Timestamp.utcnow().tz_convert("Europe/Madrid")
        change_ref_lbl = ""
        
    change_pct = ((current_price - prev_ref_price) / max(prev_ref_price, 1e-12)) * 100
    if change_ref_lbl:
        change_str = (f"▲ {change_pct:.2f}% ({change_ref_lbl})" if change_pct >= 0 else f"▼ {abs(change_pct):.2f}% ({change_ref_lbl})")
    else:
        change_str = f"▲ {change_pct:.2f}%" if change_pct >= 0 else f"▼ {abs(change_pct):.2f}%"
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    change_style = {"textAlign": "right", "marginTop": "15px", "fontWeight": "bold", "color": change_color, "fontSize": "1.1rem"}
    
    pred_vals = []
    for h in range(1, 5):
        val = preds.get(f"{h}h", current_price)
        pred_vals.append(val)
        
    pred_ui = []
    prev_val = current_price
    for i, val in enumerate(pred_vals):
        hour_diff = val - prev_val
        if hour_diff > 0:
            color = "#10b981"
            indicator = "▲"
        elif hour_diff < 0:
            color = "#ef4444"
            indicator = "▼"
        else:
            color = "#f59e0b"
            indicator = "▬"
            
        pred_ui.append(html.Div([
            html.Span(f"+{i+1}h", style={"color": "#94a3b8", "fontSize": "1.1rem"}),
            html.Span(format_price(val, currency), style={"color": "white", "fontSize": "1.2rem", "fontWeight": "bold"}),
            html.Span(f"{indicator}", style={"color": color, "fontSize": "1.2rem"})
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center", 
            "background": "rgba(255,255,255,0.03)", "padding": "15px", "borderRadius": "8px",
            "border": "1px solid rgba(255,255,255,0.05)"
        }))
        prev_val = val

    # Banda min/max alineada con la trayectoria (misma lógica para gráfico y franja)
    chart_start_price = float(df.iloc[-1]["close"]) if not df.empty and len(df) else float(current_price)
    fmin_bar, fmax_bar, upper_bound, lower_bound = _sanitize_forecast_band(
        chart_start_price, pred_vals, preds.get("min"), preds.get("max")
    )
    if not upper_bound:
        eps_b = max(abs(chart_start_price) * 0.002, 1e-6)
        fmin_bar = chart_start_price - eps_b
        fmax_bar = chart_start_price + eps_b
        upper_bound = [chart_start_price + eps_b * i / 4 for i in range(1, 5)]
        lower_bound = [chart_start_price - eps_b * i / 4 for i in range(1, 5)]
    pred_4h = float(pred_vals[3]) if len(pred_vals) == 4 else chart_start_price
        
    # 3. Decision RL
    decision = get_decision_rl(cripto, preds)
    accion = decision["accion"]
    conf = decision["confianza"] * 100
    
    if accion == "COMPRAR":
        box_bg = "linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.05))"
        box_border = "1px solid rgba(16, 185, 129, 0.3)"
        text_color = "#10b981"
    elif accion == "VENDER":
        box_bg = "linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.05))"
        box_border = "1px solid rgba(239, 68, 68, 0.3)"
        text_color = "#ef4444"
    else:
        box_bg = "linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.05))"
        box_border = "1px solid rgba(245, 158, 11, 0.3)"
        text_color = "#f59e0b"
        
    rl_ui = html.Div([
        html.H3(accion, style={"color": text_color, "margin": "0", "fontSize": "2.2rem", "letterSpacing": "2px"}),
        html.P(f"Confianza: {conf:.1f}%", style={"color": "#94a3b8", "marginTop": "10px", "fontSize": "1.1rem"})
    ], style={"background": box_bg, "border": box_border, "borderRadius": "12px", "padding": "40px", "textAlign": "center", "width": "100%"})
    
    # 4. Chart
    fig = go.Figure()
    
    if not df.empty:
        n_hist = min(len(df), CHART_HISTORY_HOURS)
        df_plot = df.tail(n_hist)
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['close'],
            mode='lines', name='Histórico (7 días)',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Proyección en el eje X con horas reales (+1h … +4h respecto al último dato)
        future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 5)]
        future_times_full = [last_time] + future_times
        pred_vals_full = [chart_start_price] + pred_vals
        upper_bound_full = [chart_start_price] + upper_bound
        lower_bound_full = [chart_start_price] + lower_bound
        pred_hover = ["Último cierre", "+1 h", "+2 h", "+3 h", "+4 h"]

        fig.add_trace(go.Scatter(
            x=future_times_full + future_times_full[::-1],
            y=upper_bound_full + lower_bound_full[::-1],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.18)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Umbral (API + trayectoria)'
        ))
        # Contorno suave del cono
        fig.add_trace(go.Scatter(
            x=future_times_full, y=upper_bound_full,
            mode='lines', line=dict(color='rgba(59,130,246,0.45)', width=1),
            hoverinfo="skip", showlegend=False, name=''
        ))
        fig.add_trace(go.Scatter(
            x=future_times_full, y=lower_bound_full,
            mode='lines', line=dict(color='rgba(59,130,246,0.45)', width=1),
            hoverinfo="skip", showlegend=False, name=''
        ))
        
        fig.add_trace(go.Scatter(
            x=future_times_full, y=pred_vals_full,
            mode='lines+markers', name='Predicción LSTM',
            line=dict(color='#f59e0b', width=2.5, dash='dash'),
            marker=dict(size=8, color='#f59e0b', line=dict(width=1, color='rgba(255,255,255,0.35)')),
            text=pred_hover,
            hovertemplate="<b>%{text}</b><br>%{y:,.4f}<extra></extra>",
        ))
        fig.add_vline(
            x=last_time, line_width=1, line_dash="dot",
            line_color="rgba(148,163,184,0.55)", layer="below"
        )
        
        max_pred = upper_bound[-1]
        min_pred = lower_bound[-1]
        
        rate = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["rate"]
        sym = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]
        
        fig.add_annotation(
            x=future_times[-1], y=max_pred,
            text=f"Máx: {sym}{max_pred*rate:,.0f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#10b981",
            font=dict(color="#10b981", size=11), yshift=10
        )
        fig.add_annotation(
            x=future_times[-1], y=min_pred,
            text=f"Mín: {sym}{min_pred*rate:,.0f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef4444",
            font=dict(color="#ef4444", size=11), yshift=-10
        )
        y_lo = min(
            float(df_plot["close"].min()),
            float(min(lower_bound)),
            float(chart_start_price),
            float(min(pred_vals)),
        ) * 0.9985
        y_hi = max(
            float(df_plot["close"].max()),
            float(max(upper_bound)),
            float(chart_start_price),
            float(max(pred_vals)),
        ) * 1.0015
        if y_hi <= y_lo:
            y_hi = y_lo * 1.002
        x_hist_min = df_plot["timestamp"].min()
        x_future_end = last_time + pd.Timedelta(hours=5)
        x_range = [x_hist_min, x_future_end]
    else:
        y_lo, y_hi = None, None
        x_range = None

    yaxis_cfg = dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
        tickprefix=CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"],
    )
    if y_lo is not None and y_hi is not None:
        yaxis_cfg["range"] = [y_lo, y_hi]

    xaxis_cfg = dict(showgrid=False, zeroline=False)
    if x_range is not None:
        xaxis_cfg["range"] = x_range

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        xaxis=xaxis_cfg,
        yaxis=yaxis_cfg,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0a1020", bordercolor="#3b82f6", font=dict(color="white"))
    )
    
    # Métricas LSTM / WF desde el mismo payload que /predict (metrics + direction_accuracy)
    metrics = get_metrica(cripto)
    rmse_val = _fmt_api_float(metrics.get("rmse_lstm"), 2)
    mae_val = _fmt_api_float(metrics.get("mae_lstm"), 2)
    mape = metrics.get("mape_lstm", 0) or 0
    try:
        mape = float(mape)
    except (TypeError, ValueError):
        mape = 0.0
    mape_val = f"{mape:.2f}%"
    dir_acc_val = _fmt_window_diracc(metrics.get("window_diracc"))

    rl_metrics = get_trained_rl_metrics(cripto) or {}
    sharpe_rl = _fmt_api_float(rl_metrics.get("sharpe"), 4)
    total_ret = rl_metrics.get("total_return")
    if total_ret is None:
        total_return_rl = "—"
    else:
        try:
            total_return_rl = f"{float(total_ret):.2f}%"
        except (TypeError, ValueError):
            total_return_rl = str(total_ret)
    calmar_rl = _fmt_api_float(rl_metrics.get("calmar_ratio"), 2)
    conf_rl = _fmt_rl_confidence(rl_metrics.get("confidence"))
    agree_rl = _fmt_signal_agreement(rl_metrics.get("signal_agreement"))
    raw_rl = _fmt_api_float(rl_metrics.get("raw_action"), 4)
    
    # 5. Franja visual: mismos límites que el cono del gráfico; marcador = predicción +4h
    rate = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["rate"]
    sym = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]
    span_bar = max(fmax_bar - fmin_bar, 1e-12)
    pos_pct = (pred_4h - fmin_bar) / span_bar * 100.0
    pos_pct = max(0.0, min(100.0, pos_pct))
    if pos_pct > 82:
        cercania_text = "Pred. +4h cerca del máx. del rango"
    elif pos_pct < 18:
        cercania_text = "Pred. +4h cerca del mín. del rango"
    else:
        cercania_text = "Pred. +4h en zona intermedia"

    franja_ui = html.Div([
        html.Div(cercania_text, style={
            "fontSize": "11px", "fontWeight": "600", "color": "#94a3b8",
            "textAlign": "center", "marginBottom": "12px", "letterSpacing": "0.02em",
        }),
        html.Div([
            html.Div([
                html.Div("Mín (rango)", style={"fontSize": "10px", "color": "#64748b", "marginBottom": "4px"}),
                html.Div(f"{sym}{fmin_bar * rate:,.0f}", style={"fontSize": "13px", "color": "#f87171", "fontWeight": "700"}),
            ], style={"flex": "1", "textAlign": "left"}),
            html.Div([
                html.Div("Predicción +4h", style={"fontSize": "10px", "color": "#64748b", "marginBottom": "4px"}),
                html.Div(format_price(pred_4h, currency), style={"fontSize": "15px", "color": "#fbbf24", "fontWeight": "800"}),
            ], style={"flex": "1", "textAlign": "center"}),
            html.Div([
                html.Div("Máx (rango)", style={"fontSize": "10px", "color": "#64748b", "marginBottom": "4px"}),
                html.Div(f"{sym}{fmax_bar * rate:,.0f}", style={"fontSize": "13px", "color": "#4ade80", "fontWeight": "700"}),
            ], style={"flex": "1", "textAlign": "right"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-end", "marginBottom": "10px"}),
        html.Div([
            html.Div(style={
                "height": "10px", "width": "100%",
                "background": "linear-gradient(90deg, rgba(248,113,113,0.35) 0%, rgba(59,130,246,0.25) 50%, rgba(74,222,128,0.35) 100%)",
                "borderRadius": "5px", "position": "relative", "border": "1px solid rgba(255,255,255,0.08)",
            }, children=[
                html.Div(style={
                    "position": "absolute", "top": "-7px", "left": f"calc({pos_pct}% - 8px)", "width": "16px", "height": "16px",
                    "background": "linear-gradient(145deg, #fff, #e2e8f0)", "borderRadius": "50%",
                    "border": "2px solid #f59e0b", "boxShadow": "0 2px 12px rgba(0,0,0,0.35)",
                    "transition": "left 0.6s ease-out", "zIndex": "3",
                }),
            ]),
        ], style={"padding": "4px 0 8px 0"}),
        html.Div([
            html.Span("0%", style={"fontSize": "9px", "color": "#475569"}),
            html.Span("posición de +4h en el rango API (alineado con el cono)", style={"fontSize": "9px", "color": "#475569", "fontStyle": "italic"}),
            html.Span("100%", style={"fontSize": "9px", "color": "#475569"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginTop": "2px"}),
    ], style={"marginTop": "8px"})

    return (
        title_str, format_price(current_price, currency), change_str, change_style, fig, pred_ui, rl_ui, 
        rmse_val, mae_val, mape_val, dir_acc_val,
        sharpe_rl, total_return_rl, calmar_rl, conf_rl, agree_rl, raw_rl,
        franja_ui
    )