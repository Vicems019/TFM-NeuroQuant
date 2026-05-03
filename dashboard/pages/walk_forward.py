import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import get_walk_forward_data


dash.register_page(__name__, path="/walk-forward", name="Walk-forward")

layout = html.Div([
    html.Div([
        html.Div([
            html.Div("Crypto LSTM + RL", className="brand-title"),
            html.Div("Dashboard TFM", className="brand-sub"),
        ], className="brand"),
        html.Div([
            dcc.Link("Visión general", href="/",             className="nav-tab"),
            dcc.Link("Backtesting",    href="/backtesting",  className="nav-tab"),
            dcc.Link("Walk-forward",   href="/walk-forward", className="nav-tab nav-tab-active"),
        ], className="nav-tabs"),
        html.Div("8 ventanas · 2023-2024", className="timestamp"),
    ], className="header"),

    html.Div([
        html.Div([
            html.Div("Retorno por ventana temporal — RL PPO vs Buy & Hold", className="chart-title"),
            dcc.Graph(id="grafico-wf", config={"displayModeBar": False}, className="chart-graph"),
        ], className="chart-container chart-full"),
    ], className="main-row"),

    html.Div([
        html.Div([
            html.Div("Detalle por ventana", className="panel-title"),
            html.Div(id="tabla-wf", className="wf-tabla"),
        ], className="chart-container chart-full"),
    ], className="main-row"),

], className="page-container")


@callback(
    Output("grafico-wf", "figure"),
    Output("tabla-wf", "children"),
    Input("store-cripto", "data"),
)
def render_wf(cripto):
    ventanas = get_walk_forward_data(cripto)
    labels = [f"V{v['ventana']}" for v in ventanas]
    rl_vals = [v["retorno_rl"] for v in ventanas]
    bh_vals = [v["retorno_bh"] for v in ventanas]

    colors_rl = ["#2ecc71" if x >= 0 else "#e74c3c" for x in rl_vals]
    colors_bh = ["rgba(74,144,217,0.7)" if x >= 0 else "rgba(231,76,60,0.5)" for x in bh_vals]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=rl_vals, name="RL PPO",
        marker_color=colors_rl, marker_line_width=0,
        hovertemplate="RL: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=labels, y=bh_vals, name="Buy & Hold",
        marker_color=colors_bh, marker_line_width=0,
        hovertemplate="B&H: <b>%{y:.2f}%</b><extra></extra>"
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_width=1)
    fig.update_layout(
        paper_bgcolor="transparent", plot_bgcolor="transparent",
        barmode="group", bargap=0.25, bargroupgap=0.05,
        font=dict(color="#aaa", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.06, bgcolor="transparent"),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#888")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, tickfont=dict(color="#666"), ticksuffix="%"),
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#D4AF37", font=dict(color="white"))
    )

    # Tabla de detalle
    header = html.Div([
        html.Span("Ventana",     className="wf-th"),
        html.Span("Período",     className="wf-th"),
        html.Span("Ret. RL",     className="wf-th wf-right"),
        html.Span("Ret. B&H",    className="wf-th wf-right"),
        html.Span("Sharpe",      className="wf-th wf-right"),
        html.Span("Win Rate",    className="wf-th wf-right"),
        html.Span("Operaciones", className="wf-th wf-right"),
    ], className="wf-row wf-header")

    filas = [header]
    for v in ventanas:
        rl_pos = v["retorno_rl"] >= 0
        bh_pos = v["retorno_bh"] >= 0
        filas.append(html.Div([
            html.Span(f"V{v['ventana']}", className="wf-td wf-bold"),
            html.Span(f"{v['inicio']} → {v['fin']}", className="wf-td wf-small"),
            html.Span(f"{'+'if rl_pos else ''}{v['retorno_rl']}%",
                      className=f"wf-td wf-right {'wf-green' if rl_pos else 'wf-red'}"),
            html.Span(f"{'+'if bh_pos else ''}{v['retorno_bh']}%",
                      className=f"wf-td wf-right {'wf-green' if bh_pos else 'wf-red'}"),
            html.Span(f"{v['sharpe']}", className="wf-td wf-right"),
            html.Span(f"{v['win_rate']:.0%}", className="wf-td wf-right"),
            html.Span(f"{v['operaciones']}", className="wf-td wf-right"),
        ], className="wf-row"))

    return fig, filas
