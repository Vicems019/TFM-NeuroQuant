import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import get_historico
from pages.lstm_utils import get_predicciones_lstm_real, format_price
dash.register_page(__name__, path="/chart-detail", name="Gráfico detallado")
CRIPTOS = ["BTC", "ETH", "SOL", "AVAX"]
RANGOS  = {"1d": 1, "3d": 3, "7d": 7, "14d": 14, "1m": 30}
layout = html.Div([
    html.Div([
        html.Div([
            dcc.Link("← Volver al dashboard", href="/", className="back-link"),
            html.Div("Vista ampliada · Gráfico de precios", className="page-hero-title"),
        ], className="detail-header"),
        # Controles
        html.Div([
            html.Div([
                html.Label("Criptomoneda", className="sett-label"),
                dcc.Dropdown(
                    id="detail-cripto-dd",
                    options=[{"label": c, "value": c} for c in CRIPTOS],
                    value="BTC", clearable=False, className="modal-dropdown",
                    style={"minWidth": "120px"}
                ),
            ], className="sett-field"),
            html.Div([
                html.Label("Rango temporal", className="sett-label"),
                dcc.Dropdown(
                    id="detail-rango-dd",
                    options=[{"label": k, "value": k} for k in RANGOS],
                    value="7d", clearable=False, className="modal-dropdown",
                    style={"minWidth": "100px"}
                ),
            ], className="sett-field"),
        ], className="sett-fields-grid"),
        # Full-size chart
        html.Div([
            dcc.Graph(id="detail-graph", config={"displayModeBar": True},
                      style={"height": "520px"}),
        ], className="panel-card", style={"marginTop": "16px"}),
    ], className="page-container"),
])
@callback(
    Output("detail-graph", "figure"),
    Input("detail-cripto-dd", "value"),
    Input("detail-rango-dd",  "value"),
    Input("store-cripto",     "data"),
)
def render_detail(dd_cripto, rango, store_cripto):
    cripto = dd_cripto or store_cripto or "BTC"
    dias   = RANGOS.get(rango, 7)
    df     = get_historico(cripto, dias)
    preds  = get_predicciones_lstm_real(cripto)
    from datetime import datetime, timedelta
    ahora  = datetime.utcnow()
    fechas_pred  = [ahora + timedelta(hours=h) for h in [1, 2, 3, 4]]
    precios_pred = [preds.get(f"{h}h", df["close"].iloc[-1]) for h in [1, 2, 3, 4]]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["fecha"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#10b981", decreasing_line_color="#ef4444",
        name="OHLCV",
    ))
    x_pred = [df["fecha"].iloc[-1], *fechas_pred]
    y_pred = [df["close"].iloc[-1], *precios_pred]
    color_pred = "#10b981" if precios_pred[-1] >= df["close"].iloc[-1] else "#ef4444"
    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        line=dict(color=color_pred, width=2, dash="dot"),
        mode="lines+markers", marker=dict(size=5),
        name="Predicción LSTM",
    ))
    fig.add_vline(x=ahora, line_width=1, line_dash="dash",
                  line_color="rgba(255,255,255,0.2)")
    fig.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font=dict(color="#94a3b8", size=11),
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", y=1.06, bgcolor="transparent"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748b"),
                   rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, tickfont=dict(color="#64748b")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#0a1020", bordercolor="#3b82f6", font=dict(color="white")),
    )
    return fig
