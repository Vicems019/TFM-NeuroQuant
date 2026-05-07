import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import sys, os
import pandas as pd
import numpy as np
from datetime import timedelta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import get_predicciones_lstm, get_decision_rl, BASE_PRICES
dash.register_page(__name__, path="/predictions", name="Predicciones IA")
def load_historical_data(coin="BTC"):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", f"{coin}_1h_raw.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values("timestamp")
            return df.tail(24 * 30) # last 30 days
    return pd.DataFrame()
# ── MODAL INFO ──────────────────────────────────────────────
modal_info = html.Div([
    html.Div(id="modal-backdrop-info", className="modal-backdrop-custom", n_clicks=0),
    html.Div([
        html.Div([
            html.Span("ℹ️ ¿Qué es Bitcoin?", className="modal-title-text"),
            html.Button("✕", id="modal-close-x-info", className="modal-close-x", n_clicks=0),
        ], className="modal-header-row"),
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
        ], className="modal-body-custom", style={"padding": "25px"}),
    ], className="modal-panel"),
], id="modal-wrapper-info", style={"display": "none"})
# ── LAYOUT ──────────────────────────────────────────────────
layout = html.Div([
    modal_info,
    
    # Header: Title + Info Button
    html.Div([
        html.Span("Predicciones IA", id="page-title-btc", className="section-title", style={"fontSize": "1.5rem"}),
        html.Button("ℹ️ Info", id="btn-info-btc", className="btn-add-op", n_clicks=0)
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}),
    
    # Top Row: 2 Horizontal Cards
    html.Div([
        # Card 1: Current Price & 24h Variation
        html.Div([
            html.Div("Precio Actual", className="pnl-label"),
            html.Div(id="card-current-price", className="pnl-val", style={"fontSize": "2rem"}),
            html.Div(id="card-price-change", style={"textAlign": "right", "marginTop": "10px", "fontWeight": "bold"})
        ], className="pnl-stat-card", style={"flex": "1", "position": "relative", "padding": "25px"}),
        
        # Card 2: Clickable Metrics + Explainability
        dcc.Link(
            html.Div([
                html.Div([
                    html.Span("Métricas y Explicabilidad", className="pnl-label", style={"fontSize": "1.1rem"}),
                    html.Span("➔", style={"color": "#3b82f6", "fontSize": "1.5rem"}),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                html.P("Explora la interpretación de los modelos, importancia de características y métricas de error.", 
                       style={"color": "#94a3b8", "marginTop": "15px", "lineHeight": "1.5"})
            ], className="pnl-stat-card", style={"flex": "1", "cursor": "pointer", "padding": "25px", "transition": "transform 0.2s"}),
            href="/metrics-explainability",
            style={"textDecoration": "none", "flex": "1", "display": "flex"}
        )
    ], style={"display": "flex", "gap": "20px", "marginBottom": "25px"}),
    
    # Main Row: Left Chart, Right Predictions
    html.Div([
        # Left: Chart with Uncertainty Cone
        html.Div([
            html.Div("Proyección de Precios a 4 Horas", className="section-title", style={"marginBottom": "15px"}),
            dcc.Graph(id="grafico-prediccion-cono", config={"displayModeBar": False}, style={"height": "500px"})
        ], className="panel-bar-chart", style={"flex": "2.2", "padding": "25px"}),
        
        # Right: Predictions Column
        html.Div([
            html.Span("Predicciones (LSTM)", className="section-title"),
            html.Div(id="lista-predicciones-4h", style={"marginTop": "20px", "display": "flex", "flexDirection": "column", "gap": "15px"}),
            
            html.Div(style={"height": "30px"}),
            
            html.Span("Recomendación (RL SAC)", className="section-title"),
            html.Div(id="caja-decision-rl", style={"marginTop": "20px", "flex": "1", "display": "flex", "alignItems": "center", "justifyContent": "center"})
        ], className="panel-ops-table", style={"flex": "1", "display": "flex", "flexDirection": "column", "padding": "25px"})
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
    Output("page-title-btc", "children"),
    Output("card-current-price", "children"),
    Output("card-price-change", "children"),
    Output("card-price-change", "style"),
    Output("grafico-prediccion-cono", "figure"),
    Output("lista-predicciones-4h", "children"),
    Output("caja-decision-rl", "children"),
    Input("store-cripto", "data"),
)
def update_predictions_page(cripto):
    if not cripto or cripto == "ALL":
        cripto = "BTC"
        
    # Set Title
    names = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "AVAX": "Avalanche", "XRP": "Ripple"}
    name = names.get(cripto, cripto)
    title_str = f"{name} ({cripto}) - Predicciones IA"
    # 1. Load Data for Card 1
    df = load_historical_data(cripto)
    if not df.empty and len(df) >= 25:
        current_price = df.iloc[-1]['close']
        prev_day_price = df.iloc[-25]['close']
        last_time = df.iloc[-1]['timestamp']
    else:
        # Fallback to mock
        current_price = BASE_PRICES["BTC"]
        prev_day_price = current_price * 0.98
        last_time = pd.Timestamp.utcnow()
        
    change_pct = ((current_price - prev_day_price) / prev_day_price) * 100
    change_str = f"▲ {change_pct:.2f}% (vs ayer)" if change_pct >= 0 else f"▼ {abs(change_pct):.2f}% (vs ayer)"
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    change_style = {"textAlign": "right", "marginTop": "15px", "fontWeight": "bold", "color": change_color, "fontSize": "1.1rem"}
    
    # 2. Get Predictions
    preds = get_predicciones_lstm(cripto)
    
    # Update predictions to align with the actual loaded current_price if possible
    # We will scale mock predictions so they start exactly at current_price
    mock_current = preds["precio_actual"]
    scale = current_price / mock_current
    
    pred_vals = []
    for h in range(1, 5):
        val = preds[f"{h}h"] * scale
        pred_vals.append(val)
        
    # Generate prediction UI rows
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
            html.Span(f"${val:,.2f}", style={"color": "white", "fontSize": "1.2rem", "fontWeight": "bold"}),
            html.Span(f"{indicator}", style={"color": color, "fontSize": "1.2rem"})
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center", 
            "background": "rgba(255,255,255,0.03)", "padding": "15px", "borderRadius": "8px",
            "border": "1px solid rgba(255,255,255,0.05)"
        }))
        prev_val = val
        
    # 3. Get Decision RL
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
        html.H3(accion, style={"color": text_color, "margin": "0", "fontSize": "2rem", "letterSpacing": "2px"}),
        html.P(f"Confianza: {conf:.1f}%", style={"color": "#94a3b8", "marginTop": "10px", "fontSize": "1.1rem"})
    ], style={"background": box_bg, "border": box_border, "borderRadius": "12px", "padding": "30px", "textAlign": "center", "width": "100%"})
    
    # 4. Create Chart with Cone
    fig = go.Figure()
    
    if not df.empty:
        # Plot last 48 hours for better visibility of the new prediction
        df_plot = df.tail(48)
        fig.add_trace(go.Scatter(
            x=df_plot['timestamp'], y=df_plot['close'],
            mode='lines', name='Histórico',
            line=dict(color='#3b82f6', width=2)
        ))
        
        # Cone prediction
        future_times = [last_time + pd.Timedelta(hours=i) for i in range(1, 5)]
        
        # Max and Min paths for the cone (synthetic expansion of uncertainty)
        # alpha is the cone expansion factor
        alpha = 0.005 
        upper_bound = [v * (1 + alpha * (i+1)) for i, v in enumerate(pred_vals)]
        lower_bound = [v * (1 - alpha * (i+1)) for i, v in enumerate(pred_vals)]
        
        # Extend from the current point
        future_times_full = [last_time] + future_times
        pred_vals_full = [current_price] + pred_vals
        upper_bound_full = [current_price] + upper_bound
        lower_bound_full = [current_price] + lower_bound
        
        # Add cone
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
        
        # Add prediction line
        fig.add_trace(go.Scatter(
            x=future_times_full, y=pred_vals_full,
            mode='lines+markers', name='Predicción',
            line=dict(color='#f59e0b', width=2, dash='dash'),
            marker=dict(size=6, color='#f59e0b')
        ))
        
        # Annotations for max and min
        max_pred = max(upper_bound)
        min_pred = min(lower_bound)
        
        fig.add_annotation(
            x=future_times[-1], y=max_pred,
            text=f"Máx: ${max_pred:,.0f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#10b981",
            font=dict(color="#10b981", size=11), yshift=10
        )
        fig.add_annotation(
            x=future_times[-1], y=min_pred,
            text=f"Mín: ${min_pred:,.0f}",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#ef4444",
            font=dict(color="#ef4444", size=11), yshift=-10
        )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=10, b=10),
        font=dict(color="#94a3b8"),
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickprefix="$"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0a1020", bordercolor="#3b82f6", font=dict(color="white"))
    )
    
    return title_str, f"${current_price:,.2f}", change_str, change_style, fig, pred_ui, rl_ui