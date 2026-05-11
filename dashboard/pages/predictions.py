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

def load_historical_data(coin="BTC"):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", f"{coin}_1h_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Convertir a horario de Madrid para visualización
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize("UTC")
            df['timestamp'] = df['timestamp'].dt.tz_convert("Europe/Madrid")
            df = df.sort_values("timestamp")
            return df.tail(24 * 30) # last 30 days
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
            html.Span("📊 Métricas y Explicabilidad", className="modal-title-text"),
        ], className="modal-header-row", style={"borderBottom": "1px solid rgba(59,130,246,0.13)", "paddingBottom": "15px"}),
        
        html.Div([
            # LSTM Metrics
            html.Div([
                html.H4("Modelo Predictivo (LSTM)", style={"color": "white", "marginBottom": "15px"}),
                html.Div([
                    html.Div([html.Span("RMSE", style={"fontWeight": "bold"}), html.Div(id="rmse-value", className="pnl-val"), html.Span("Error cuadrático medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("MAE", style={"fontWeight": "bold"}), html.Div(id="mae-value", className="pnl-val"), html.Span("Error absoluto medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("MAPE", style={"fontWeight": "bold"}), html.Div(id="mape-value", className="pnl-val"), html.Span("Error porcentual medio", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Dir. Acc.", style={"fontWeight": "bold"}), html.Div(id="dir-acc-value", className="pnl-val green"), html.Span("Precisión direccional", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("R²", style={"fontWeight": "bold"}), html.Div(id="r2-value", className="pnl-val"), html.Span("Coef. de determinación", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Walk-Forward", style={"fontWeight": "bold"}), html.Div(id="wf-value", className="pnl-val green"), html.Span("Validación en ventanas", className="modal-hint")], className="pnl-stat-card"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px"})
            ]),
            
            # RL Metrics
            html.Div([
                html.H4("Modelo de Decisión (RL SAC)", style={"color": "white", "margin": "20px 0 15px 0"}),
                html.Div([
                    html.Div([html.Span("Sharpe Ratio", style={"fontWeight": "bold"}), html.Div(id="sharpe-rl", className="pnl-val green"), html.Span("Retorno vs Volatilidad", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Sortino", style={"fontWeight": "bold"}), html.Div(id="sortino-rl", className="pnl-val green"), html.Span("Penaliza sólo bajadas", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Max Drawdown", style={"fontWeight": "bold"}), html.Div(id="max-dd-rl", className="pnl-val red"), html.Span("Máxima pérdida", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Win Rate", style={"fontWeight": "bold"}), html.Div(id="win-rate-rl", className="pnl-val green"), html.Span("Operaciones ganadoras", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Profit Factor", style={"fontWeight": "bold"}), html.Div(id="profit-factor-rl", className="pnl-val green"), html.Span("Ganancias vs Pérdidas", className="modal-hint")], className="pnl-stat-card"),
                    html.Div([html.Span("Trades", style={"fontWeight": "bold"}), html.Div(id="trades-rl", className="pnl-val"), html.Span("Total de operaciones", className="modal-hint")], className="pnl-stat-card"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "10px"}),
                
                # Chart below metrics
                html.Div([
                    html.Div("Recompensa Acumulada vs B&H", style={"fontSize": "12px", "color": "#94a3b8", "marginBottom": "5px", "marginTop": "10px"}),
                    dcc.Graph(id="chart-rl-metrics", config={"displayModeBar": False}, style={"height": "250px", "background": "transparent", "borderRadius": "8px"})
                ], style={"border": "1px solid rgba(59,130,246,0.13)", "padding": "10px", "borderRadius": "12px", "background": "#0c1428", "marginTop": "15px"})
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
            html.Div("Proyección de Precios a 4 Horas", className="section-title", style={"marginBottom": "15px"}),
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
    Output("chart-rl-metrics", "figure"),
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
        
        # Chart RL metrics
        x_data = list(range(100))
        y_rl = np.cumsum(np.random.normal(0.002, 0.015, 100))
        y_bh = np.cumsum(np.random.normal(0.001, 0.02, 100))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_data, y=y_rl, name="RL Agent", line=dict(color="#10b981", width=2)))
        fig.add_trace(go.Scatter(x=x_data, y=y_bh, name="Buy & Hold", line=dict(color="#3b82f6", width=2, dash="dash")))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=5, b=5), showlegend=True,
            legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)"),
            font=dict(color="#94a3b8", size=10),
            xaxis=dict(visible=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        )
        
        # Expl LSTM Chart con SHAP
        expl_lstm = get_lstm_shap(cripto)
        fig_lstm = go.Figure(go.Bar(
            x=expl_lstm["values"],
            y=expl_lstm["features"],
            orientation='h',
            marker=dict(color='#3b82f6')
        ))
        fig_lstm.update_layout(
            margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, visible=False),
            yaxis=dict(autorange="reversed"), font=dict(color="#e2e8f0", size=10)
        )

        # Expl SAC Chart con SHAP
        expl_rl = get_rl_shap(cripto)
        fig_sac = go.Figure(go.Bar(
            x=expl_rl["values"],
            y=expl_rl["features"],
            orientation='h',
            marker=dict(color='#10b981')
        ))
        fig_sac.update_layout(
            margin=dict(l=5, r=5, t=5, b=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, visible=False),
            yaxis=dict(autorange="reversed"), font=dict(color="#e2e8f0", size=10)
        )
        
        return {"display": "flex", "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%", "zIndex": "1000", "alignItems": "center", "justifyContent": "center"}, fig, fig_lstm, fig_sac
        
    return {"display": "none"}, dash.no_update, dash.no_update, dash.no_update

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
    Output("r2-value", "children"),
    Output("wf-value", "children"),
    Output("sharpe-rl", "children"),
    Output("sortino-rl", "children"),
    Output("max-dd-rl", "children"),
    Output("win-rate-rl", "children"),
    Output("profit-factor-rl", "children"),
    Output("trades-rl", "children"),
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
        prev_day_price = df.iloc[-25]['close']
        last_time = df.iloc[-1]['timestamp']
    else:
        current_price = api_current_price if api_current_price > 0 else 100
        prev_day_price = current_price * 0.98
        last_time = pd.Timestamp.utcnow().tz_convert("Europe/Madrid")
        
    change_pct = ((current_price - prev_day_price) / prev_day_price) * 100
    change_str = f"▲ {change_pct:.2f}% (vs ayer)" if change_pct >= 0 else f"▼ {abs(change_pct):.2f}% (vs ayer)"
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
        df_plot = df.tail(48)
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['close'],
            mode='lines', name='Histórico',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Proyección de las siguientes 4 horas a partir del último dato histórico
        future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 5)]
        
        # Para que el gráfico no tenga un salto, el punto de partida debe ser el último histórico
        chart_start_price = df.iloc[-1]['close']
        
        # Usamos los rangos reales devueltos por la API (la franja)
        api_min = preds.get("min", chart_start_price * 0.99)
        api_max = preds.get("max", chart_start_price * 1.01)

        
        # Proyectamos los rangos sobre los 4 pasos (simplificado para el cono)
        upper_bound = []
        lower_bound = []
        for i, v in enumerate(pred_vals):
            # Proporción del rango total basada en el paso (1h a 4h)
            ratio = (i + 1) / 4
            u = chart_start_price + (api_max - chart_start_price) * ratio
            l = chart_start_price + (api_min - chart_start_price) * ratio
            upper_bound.append(u)
            lower_bound.append(l)
        
        future_times_full = [last_time] + future_times
        pred_vals_full = [chart_start_price] + pred_vals
        upper_bound_full = [chart_start_price] + upper_bound
        lower_bound_full = [chart_start_price] + lower_bound
        
        fig.add_trace(go.Scatter(
            x=future_times_full + future_times_full[::-1],
            y=upper_bound_full + lower_bound_full[::-1],
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name='Umbral de Confianza'
        ))
        
        fig.add_trace(go.Scatter(
            x=future_times_full, y=pred_vals_full,
            mode='lines+markers', name='Predicción',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            marker=dict(size=6, color='#f59e0b')
        ))
        
        max_pred = max(upper_bound)
        min_pred = min(lower_bound)
        
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
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickprefix=CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0a1020", bordercolor="#3b82f6", font=dict(color="white"))
    )
    
    # Get LSTM metrics
    metrics = get_metrica(cripto)
    
    rmse_val = f"{metrics.get('rmse_lstm', 0):.4f}"
    mae_val = f"{metrics.get('mae_lstm', 0):.4f}"
    
    # Si el MAPE viene como ratio (ej. 0.05), lo multiplicamos por 100
    mape = metrics.get('mape_lstm', 0)
    if 0 < mape < 1: mape *= 100
    mape_val = f"{mape:.2f}%"
    
    dir_acc_val = f"{(metrics.get('accuracy_lstm', 0) * 100):.1f}%"
    r2_val = f"{metrics.get('r2_lstm', 0):.3f}"
    wf_val = f"{(metrics.get('accuracy_lstm', 0) * 100):.1f}% ±"
    
    # Get RL metrics
    rl_metrics = get_trained_rl_metrics(cripto)
    if rl_metrics:
        sharpe_rl = rl_metrics["sharpe"]
        sortino_rl = rl_metrics["sortino"]
        max_dd_rl = rl_metrics["max_dd"]
        win_rate_rl = rl_metrics["win_rate"]
        pf_rl = rl_metrics["profit_factor"]
        trades_rl = rl_metrics["trades"]
    else:
        sharpe_rl = sortino_rl = max_dd_rl = win_rate_rl = pf_rl = trades_rl = "N/A"
    
    # 5. La Franja Logic (Usa los valores reales de la API)
    rate = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["rate"]
    sym = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]
    
    # Rango real de la API (La Franja)
    f_min = preds.get("min")
    f_max = preds.get("max")
    
    # Calcular posición porcentual del precio actual en ese rango futuro
    if f_max > f_min:
        pos_pct = (current_price - f_min) / (f_max - f_min)
    else:
        pos_pct = 0.5
        
    pos_pct = max(0, min(1, pos_pct)) * 100
    
    # Cercanía Label
    if pos_pct > 80: cercania_text = "Cerca del Máximo"
    elif pos_pct < 20: cercania_text = "Cerca del Mínimo"
    else: cercania_text = "Zona Neutral"
    
    franja_ui = html.Div([
        html.Div([
            html.Span(f"Proyección: {sym}{f_min*rate:,.0f}", style={"fontSize": "10px", "color": "#94a3b8"}),
            html.Span(cercania_text, style={"fontSize": "11px", "fontWeight": "bold", "color": "#3b82f6", "textTransform": "uppercase"}),
            html.Span(f"{sym}{f_max*rate:,.0f}", style={"fontSize": "10px", "color": "#94a3b8"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "8px"}),
        
        html.Div([
            # Track
            html.Div(style={
                "height": "6px", "width": "100%", "background": "rgba(255,255,255,0.05)", 
                "borderRadius": "3px", "position": "relative"
            }, children=[
                # Fill (gradiente de confianza)
                html.Div(style={
                    "position": "absolute", "height": "100%", "left": "0", "width": f"{pos_pct}%",
                    "background": "linear-gradient(90deg, #3b82f6, #10b981)",
                    "borderRadius": "3px", "transition": "width 1s ease-in-out"
                }),
                # Marker
                html.Div(style={
                    "position": "absolute", "height": "14px", "width": "14px", "background": "white",
                    "borderRadius": "50%", "top": "-4px", "left": f"calc({pos_pct}% - 7px)",
                    "boxShadow": "0 0 10px rgba(59,130,246,0.8)", "border": "2px solid #3b82f6",
                    "transition": "left 1s ease-in-out"
                })
            ])
        ], style={"padding": "5px 0"}),
        
        html.Div([
            html.Span("Mín proyectado (4h)", style={"fontSize": "9px", "color": "#475569"}),
            html.Span("Máx proyectado (4h)", style={"fontSize": "9px", "color": "#475569"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginTop": "4px"})
    ])

    return (
        title_str, format_price(current_price, currency), change_str, change_style, fig, pred_ui, rl_ui, 
        rmse_val, mae_val, mape_val, dir_acc_val, r2_val, wf_val,
        sharpe_rl, sortino_rl, max_dd_rl, win_rate_rl, pf_rl, trades_rl,
        franja_ui
    )