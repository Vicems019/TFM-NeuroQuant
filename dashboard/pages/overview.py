import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import (
    get_predicciones_lstm, get_decision_rl, get_historico, get_metricas
)

dash.register_page(__name__, path="/", name="Visión general")

CRIPTOS = ["BTC", "ETH", "SOL", "XRP", "AVAX"]
RANGOS = {"1d": 1, "3d": 3, "7d": 7, "1m": 30}

# ── Layout ──────────────────────────────────────────────────────────────────
layout = html.Div([

    # ── HEADER ──
    html.Div([
        html.Div([
            html.Div("Crypto LSTM + RL", className="brand-title"),
            html.Div("Dashboard TFM", className="brand-sub"),
        ], className="brand"),
        html.Div([
            dcc.Link("Visión general", href="/",         className="nav-tab nav-tab-active"),
            dcc.Link("Backtesting",    href="/backtesting", className="nav-tab"),
            dcc.Link("Walk-forward",   href="/walk-forward", className="nav-tab"),
        ], className="nav-tabs"),
        html.Div(id="timestamp-header", className="timestamp"),
    ], className="header"),

    # ── SELECTOR CRIPTO ──
    html.Div([
        html.Button(c, id={"type": "btn-cripto", "index": c},
                    className="btn-cripto btn-cripto-active" if c == "BTC" else "btn-cripto")
        for c in CRIPTOS
    ], className="cripto-selector"),

    # ── KPI CARDS ──
    html.Div([
        html.Div([html.Div("Precio actual", className="kpi-label"),
                  html.Div(id="kpi-precio", className="kpi-value kpi-precio"),
                  html.Div(id="kpi-cambio", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Retorno RL (eval)", className="kpi-label"),
                  html.Div(id="kpi-retorno-rl", className="kpi-value kpi-green"),
                  html.Div(id="kpi-vs-bh", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Sharpe real", className="kpi-label"),
                  html.Div(id="kpi-sharpe", className="kpi-value"),
                  html.Div("datos nunca vistos", className="kpi-sub")], className="kpi-card"),
        html.Div([html.Div("Max drawdown", className="kpi-label"),
                  html.Div(id="kpi-dd", className="kpi-value kpi-red"),
                  html.Div("eval final", className="kpi-sub")], className="kpi-card"),
    ], className="kpi-row"),

    # ── FILA PRINCIPAL ──
    html.Div([

        # Gráfico
        html.Div([
            html.Div([
                html.Span("Precio histórico + predicciones LSTM", className="chart-title"),
                dcc.Dropdown(
                    id="dd-rango",
                    options=[{"label": k, "value": k} for k in RANGOS],
                    value="7d",
                    clearable=False,
                    className="rango-dropdown"
                ),
            ], className="chart-header"),
            dcc.Graph(id="grafico-precio", config={"displayModeBar": False},
                      className="chart-graph"),
        ], className="chart-container"),

        # Panel predicciones
        html.Div([
            html.Div("Predicciones LSTM — próximas 4h", className="panel-title"),
            html.Div(id="tabla-predicciones", className="pred-tabla"),
            html.Div(id="panel-decision", className="decision-panel"),
            html.Button([
                "Actualizar predicción ",
                html.Span("↗", style={"fontSize": "1rem"})
            ], id="btn-actualizar", className="btn-actualizar", n_clicks=0),
        ], className="predictions-panel"),

    ], className="main-row"),

], className="page-container")


# ── CALLBACKS ────────────────────────────────────────────────────────────────

@callback(
    Output("store-cripto", "data"),
    [Input({"type": "btn-cripto", "index": c}, "n_clicks") for c in CRIPTOS],
    prevent_initial_call=True
)
def cambiar_cripto(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "BTC"
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    return json.loads(btn_id)["index"]


@callback(
    Output("store-predicciones", "data"),
    Output("store-decision", "data"),
    Input("btn-actualizar", "n_clicks"),
    Input("intervalo-auto", "n_intervals"),
    Input("store-cripto", "data"),
)
def actualizar_predicciones(n_clicks, n_intervals, cripto):
    preds = get_predicciones_lstm(cripto)
    decision = get_decision_rl(cripto, preds)
    return preds, decision


@callback(
    Output("kpi-precio", "children"),
    Output("kpi-cambio", "children"),
    Output("kpi-cambio", "className"),
    Output("kpi-retorno-rl", "children"),
    Output("kpi-vs-bh", "children"),
    Output("kpi-sharpe", "children"),
    Output("kpi-dd", "children"),
    Output("timestamp-header", "children"),
    Input("store-predicciones", "data"),
    Input("store-cripto", "data"),
)
def actualizar_kpis(preds, cripto):
    from datetime import datetime
    metricas = get_metricas(cripto)
    precio = preds.get("precio_actual", 0)
    cambio = preds.get("cambio_24h", 0)

    # Formateo dinámico según precio
    if precio >= 1000:
        fmt_precio = f"${precio:,.0f}"
    elif precio >= 1:
        fmt_precio = f"${precio:,.3f}"
    else:
        fmt_precio = f"${precio:.4f}"

    color_cambio = "kpi-sub kpi-green" if cambio >= 0 else "kpi-sub kpi-red"
    sign = "+" if cambio >= 0 else ""
    ts = datetime.utcnow().strftime("Actualizado: %H:%M UTC")

    return (
        fmt_precio,
        f"{sign}{cambio:.2f}% 24h",
        color_cambio,
        f"+{metricas['retorno_rl']}%",
        f"vs B&H +{metricas['retorno_bh']}%",
        f"{metricas['sharpe']}",
        f"{metricas['max_dd']}%",
        ts
    )


@callback(
    Output("grafico-precio", "figure"),
    Input("store-cripto", "data"),
    Input("store-predicciones", "data"),
    Input("dd-rango", "value"),
)
def actualizar_grafico(cripto, preds, rango):
    dias = RANGOS.get(rango, 7)
    df = get_historico(cripto, dias)
    precio_actual = preds.get("precio_actual", df["close"].iloc[-1])

    from datetime import datetime, timedelta
    ahora = datetime.utcnow()
    fechas_pred = [ahora + timedelta(hours=h) for h in [1, 2, 3, 4]]
    precios_pred = [preds.get(f"{h}h", precio_actual) for h in [1, 2, 3, 4]]

    fig = go.Figure()

    # Área bajo la curva histórica
    fig.add_trace(go.Scatter(
        x=df["fecha"], y=df["close"],
        fill="tozeroy",
        fillcolor="rgba(212,175,55,0.06)",
        line=dict(color="#D4AF37", width=1.5),
        name="Precio histórico",
        hovertemplate="<b>%{y:,.2f}</b><br>%{x}<extra></extra>"
    ))

    # Línea de predicciones
    x_pred = [df["fecha"].iloc[-1], *fechas_pred]
    y_pred = [precio_actual, *precios_pred]
    color_pred = "#e74c3c" if precios_pred[-1] < precio_actual else "#2ecc71"

    fig.add_trace(go.Scatter(
        x=x_pred, y=y_pred,
        line=dict(color=color_pred, width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=5, color=color_pred),
        name="Predicción LSTM",
        hovertemplate="<b>LSTM: %{y:,.2f}</b><extra></extra>"
    ))

    # Línea vertical "ahora"
    fig.add_vline(x=ahora, line_width=1, line_dash="dash",
                  line_color="rgba(255,255,255,0.25)")

    fig.update_layout(
        paper_bgcolor="transparent",
        plot_bgcolor="transparent",
        font=dict(color="#aaa", size=11),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08, x=0,
                    bgcolor="transparent", font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, showline=False,
                   tickfont=dict(color="#666")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   zeroline=False, tickfont=dict(color="#666")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#D4AF37",
                        font=dict(color="white"))
    )
    return fig


@callback(
    Output("tabla-predicciones", "children"),
    Input("store-predicciones", "data"),
    Input("store-cripto", "data"),
)
def render_tabla(preds, cripto):
    if not preds:
        return html.Div("Cargando...", className="loading-text")
    precio_actual = preds.get("precio_actual", 0)
    filas = []
    for h in [1, 2, 3, 4]:
        precio = preds.get(f"{h}h", precio_actual)
        cambio = (precio - precio_actual) / precio_actual * 100 if precio_actual else 0
        es_neg = cambio < 0
        badge_cls = "pred-badge pred-badge-neg" if es_neg else "pred-badge pred-badge-pos"
        sign = "" if es_neg else "+"
        if precio >= 1000:
            fmt = f"${precio:,.0f}"
        elif precio >= 1:
            fmt = f"${precio:,.3f}"
        else:
            fmt = f"${precio:.4f}"
        filas.append(html.Div([
            html.Span(f"en {h}h", className="pred-hora"),
            html.Span(fmt, className="pred-precio"),
            html.Span(f"{sign}{cambio:.2f}%", className=badge_cls),
        ], className="pred-fila"))
    return filas


@callback(
    Output("panel-decision", "children"),
    Output("panel-decision", "className"),
    Input("store-decision", "data"),
)
def render_decision(decision):
    if not decision:
        return html.Div(), "decision-panel"
    accion = decision.get("accion", "HOLD")
    confianza = decision.get("confianza", 0)
    colores = {"COMPRAR": "decision-panel decision-buy",
               "VENDER":  "decision-panel decision-sell",
               "HOLD":    "decision-panel decision-hold"}
    iconos = {"COMPRAR": "🟢", "VENDER": "🔴", "HOLD": "⚪"}
    return html.Div([
        html.Div("Decisión RL (PPO)", className="decision-label"),
        html.Div([
            html.Span(iconos[accion], style={"marginRight": "8px"}),
            html.Span(accion, className="decision-accion"),
        ], className="decision-body"),
        html.Div(f"Confianza: {confianza:.0%}", className="decision-confianza"),
    ]), colores.get(accion, "decision-panel")
