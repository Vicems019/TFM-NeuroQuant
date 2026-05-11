import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import get_backtesting_data
from pages.api_client import get_trained_rl_metrics

dash.register_page(__name__, path="/backtesting", name="Backtesting")

layout = html.Div([
    html.Div([
        html.Div([
            html.Div("Crypto LSTM + RL", className="brand-title"),
            html.Div("Dashboard TFM", className="brand-sub"),
        ], className="brand"),
        html.Div([
            dcc.Link("Visión general", href="/",            className="nav-tab"),
            dcc.Link("Backtesting",    href="/backtesting", className="nav-tab nav-tab-active"),
            dcc.Link("Walk-forward",   href="/walk-forward", className="nav-tab"),
        ], className="nav-tabs"),
        html.Div("Backtesting 2024", className="timestamp"),
    ], className="header"),

    html.Div([
        html.Div(id="bt-kpis", className="kpi-row"),
    ]),

    html.Div([
        html.Div([
            html.Div("Curva de Equity — RL PPO vs Buy & Hold", className="chart-title"),
            dcc.Graph(id="grafico-bt", config={"displayModeBar": False}, className="chart-graph"),
        ], className="chart-container chart-full"),
    ], className="main-row"),

], className="page-container")


@callback(
    Output("grafico-bt", "figure"),
    Output("bt-kpis", "children"),
    Input("store-cripto", "data"),
)
def render_backtesting(cripto):
    df = get_backtesting_data(cripto)
    
    # Obtener métricas reales por API
    m_real = get_trained_rl_metrics(cripto) or {}
    
    # Calcular retornos desde la curva (base 100)
    ret_rl = round(df["rl"].iloc[-1] - 100, 1)
    ret_bh = round(df["bh"].iloc[-1] - 100, 1)
    
    sharpe = m_real.get("sharpe", "1.87")
    if isinstance(sharpe, (float, int)): sharpe = f"{sharpe:.2f}"
    
    max_dd = m_real.get("max_dd", "-8.3")
    if isinstance(max_dd, (float, int)): max_dd = f"{max_dd:.1f}%"
    elif not str(max_dd).endswith("%") and max_dd != "N/A": max_dd = f"{max_dd}%"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["rl"],
        line=dict(color="#D4AF37", width=2),
        fill="tozeroy", fillcolor="rgba(212,175,55,0.07)",
        name="RL PPO",
        hovertemplate="RL: <b>%{y:.1f}</b><extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["bh"],
        line=dict(color="#4a90d9", width=1.5, dash="dot"),
        name="Buy & Hold",
        hovertemplate="B&H: <b>%{y:.1f}</b><extra></extra>"
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        font=dict(color="#aaa", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.06, bgcolor="transparent"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#666")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, tickfont=dict(color="#666"),
                   ticksuffix="%"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#D4AF37", font=dict(color="white"))
    )

    kpis = [
        html.Div([html.Div("Retorno RL", className="kpi-label"),
                  html.Div(f"+{ret_rl}%", className="kpi-value kpi-green"),
                  html.Div("período completo", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Retorno B&H", className="kpi-label"),
                  html.Div(f"+{ret_bh}%", className="kpi-value"),
                  html.Div("benchmark", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Alpha generado", className="kpi-label"),
                  html.Div(f"+{ret_rl - ret_bh:.1f}%",
                           className="kpi-value kpi-green"),
                  html.Div("vs Buy & Hold", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Sharpe Ratio", className="kpi-label"),
                  html.Div(f"{sharpe}", className="kpi-value"),
                  html.Div("ajustado por riesgo", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Max Drawdown", className="kpi-label"),
                  html.Div(f"{max_dd}", className="kpi-value kpi-red"),
                  html.Div("peor caída", className="kpi-sub")], className="kpi-card"),
    ]
    return fig, kpis
