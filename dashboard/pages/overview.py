import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import (
    get_predicciones_lstm, get_decision_rl, get_historico, get_metricas,
    get_rentabilidad_periodica, get_rentabilidad_all,
    get_rentabilidad_absoluta, get_historial_operaciones,
    BASE_PRICES, CRYPTO_COLORS
)
dash.register_page(__name__, path="/", name="Visión General")
CRIPTOS    = ["ALL", "BTC", "ETH", "SOL", "AVAX"]
CARD_COINS = ["BTC", "ETH", "SOL"]
PERIODO_LABEL = {"1d": "Diaria", "7d": "Semanal", "1m": "Mensual"}
def _fmt(precio):
    if precio >= 1000: return f"${precio:,.0f}"
    if precio >= 1:    return f"${precio:,.3f}"
    return f"${precio:.4f}"
def _mini_chart(coin):
    df    = get_historico(coin, 3)
    close = df["close"].tolist()
    up    = close[-1] >= close[0]
    color = "#10b981" if up else "#ef4444"
    fill  = "rgba(16,185,129,0.12)" if up else "rgba(239,68,68,0.10)"
    fig   = go.Figure(go.Scatter(y=close, fill="tozeroy", fillcolor=fill,
                                  line=dict(color=color, width=1.8), hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
def chart_card(coin):
    preds  = get_predicciones_lstm(coin)
    precio = preds.get("precio_actual", 0)
    cambio = preds.get("cambio_24h", 0)
    cls    = "badge-green" if cambio >= 0 else "badge-red"
    sign   = "+" if cambio >= 0 else ""
    return dcc.Link(html.Div([
        html.Div([
            html.Div([
                html.Span(coin, className="card-coin-name"),
                html.Span(f"{sign}{cambio:.2f}%", className=f"card-badge {cls}"),
            ], className="card-coin-row"),
            html.Button("⛶", className="btn-expand", title="Ampliar"),
        ], className="chart-card-header"),
        html.Div(_fmt(precio), className="card-price"),
        dcc.Graph(id=f"mini-chart-{coin}", figure=_mini_chart(coin),
                  config={"displayModeBar": False, "staticPlot": True}, className="mini-chart"),
    ], className="chart-card"), href=f"/predictions?coin={coin}", style={"textDecoration": "none", "color": "inherit"})
# ── CUSTOM MODAL ─────────────────────────────────────
modal = html.Div([
    html.Div(id="modal-backdrop", className="modal-backdrop-custom", n_clicks=0),
    html.Div([
        html.Div([
            html.Span("➕ Nueva operación", className="modal-title-text"),
            html.Button("✕", id="modal-close-x", className="modal-close-x", n_clicks=0),
        ], className="modal-header-row"),
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Tipo", className="modal-label"),
                    dcc.Dropdown(id="modal-tipo",
                        options=[{"label": t, "value": t} for t in ["COMPRAR", "VENDER", "HOLD"]],
                        value="COMPRAR", clearable=False, className="modal-dropdown"),
                ], className="modal-field"),
                html.Div([
                    html.Label("Criptomoneda", className="modal-label"),
                    dcc.Dropdown(id="modal-cripto",
                        options=[{"label": c, "value": c} for c in ["BTC","ETH","SOL","AVAX"]],
                        value="BTC", clearable=False, className="modal-dropdown"),
                ], className="modal-field"),
            ], className="modal-row-2"),
            html.Div([
                html.Div([
                    html.Label("Precio de entrada ($)", className="modal-label"),
                    dcc.Input(id="modal-precio", type="number", placeholder="auto",
                              className="modal-input", debounce=False),
                ], className="modal-field"),
                html.Div([
                    html.Label("Cantidad", className="modal-label"),
                    dcc.Input(id="modal-cantidad", type="number", placeholder="ej. 0.01",
                              className="modal-input"),
                ], className="modal-field"),
            ], className="modal-row-2"),
            html.Div([
                html.Label("Margen (%)", className="modal-label"),
                html.Div([
                    dcc.Input(id="modal-margen", type="number", placeholder="ej. 10",
                              min=1, max=100, step=1, className="modal-input modal-input-sm"),
                    html.Span("× apalancamiento", className="modal-hint"),
                ], className="modal-margen-row"),
            ], className="modal-field"),
            html.Div(id="modal-price-hint", className="modal-price-hint"),
        ], className="modal-body-custom"),
        html.Div([
            html.Button("Cancelar",         id="modal-cancel",  className="btn-modal-cancel",  n_clicks=0),
            html.Button("Confirmar operación", id="modal-confirm", className="btn-modal-confirm", n_clicks=0),
        ], className="modal-footer-row"),
    ], className="modal-panel"),
], id="modal-wrapper", style={"display": "none"})
# ── LAYOUT ────────────────────────────────────────────────────────────────────
layout = html.Div([
    modal,
    # Selector ALL | BTC | ETH | SOL | AVAX
    html.Div([
        *[html.Button(c, id={"type": "btn-cripto", "index": c},
            className="btn-cripto" + (" btn-cripto-active" if c == "ALL" else ""))
          for c in CRIPTOS],
        html.Div(id="timestamp-header", className="page-timestamp"),
    ], className="cripto-selector"),
    # 3 Chart Cards
    html.Div([chart_card(c) for c in CARD_COINS], className="chart-cards-row"),
    # Bottom: Bar Chart + Table
    html.Div([
        # Left – P&L summary + bar chart
        html.Div([
            # Absolute P&L panel
            html.Div(id="pnl-summary-panel", className="pnl-summary-panel"),
            # Bar chart
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("Rentabilidad", className="section-title"),
                        html.Span(id="rentabilidad-subtitle", className="section-sub"),
                    ], className="section-header-left"),
                    dcc.Dropdown(
                        id="dd-rango",
                        options=[{"label": "1D", "value": "1d"},
                                 {"label": "7D", "value": "7d"},
                                 {"label": "1M", "value": "1m"}],
                        value="7d", clearable=False, className="rango-dropdown",
                    ),
                ], className="section-header"),
                dcc.Graph(id="grafico-rentabilidad",
                          config={"displayModeBar": False}, style={"height": "240px"}),
            ], className="bar-chart-inner"),
        ], className="panel-bar-chart"),
        # Right – Operations table
        html.Div([
            html.Div([
                html.Span("Acciones realizadas", className="section-title"),
                html.Button("＋", id="btn-add-op", className="btn-add-op",
                            n_clicks=0, title="Añadir operación manual"),
            ], className="section-header"),
            html.Div(id="tabla-operaciones", className="ops-table-wrapper"),
        ], className="panel-ops-table"),
    ], className="bottom-row"),
], className="page-container")
# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(
    Output({"type": "btn-cripto", "index": dash.ALL}, "className"),
    Input("store-cripto", "data"),
    State({"type": "btn-cripto", "index": dash.ALL}, "id"),

)
def actualizar_botones_cripto(selected_c, ids):
    if not ids:
        return dash.no_update
    if not selected_c:
        selected_c = "ALL"
    return ["btn-cripto" + (" btn-cripto-active" if id["index"] == selected_c else "") for id in ids]
@callback(
    Output("store-predicciones", "data"),
    Output("store-decision",     "data"),
    Input("intervalo-auto",      "n_intervals"),
    Input("store-cripto",        "data"),
)
def actualizar_predicciones(_, cripto):
    c     = "BTC" if cripto == "ALL" else cripto
    preds = get_predicciones_lstm(c)
    return preds, get_decision_rl(c, preds)
@callback(
    Output("timestamp-header",      "children"),
    Output("rentabilidad-subtitle", "children"),
    Input("store-cripto",           "data"),
    Input("dd-rango",               "value"),
)
def actualizar_meta(cripto, periodo):
    from datetime import datetime
    ts    = datetime.utcnow().strftime("%H:%M UTC")
    label = PERIODO_LABEL.get(periodo, "")
    sub   = f"· {cripto}  ·  {label}"
    return f"Actualizado: {ts}", sub
@callback(
    Output("grafico-rentabilidad", "figure"),
    Input("store-cripto", "data"),
    Input("dd-rango",     "value"),
)
def actualizar_bar_chart(cripto, periodo):
    fig = go.Figure()
    if cripto == "ALL":
        info = get_rentabilidad_all(periodo)
        labels = info["labels"]
        for coin, vals in info["data"].items():
            fig.add_trace(go.Bar(
                name=coin, x=labels, y=vals,
                marker_color=CRYPTO_COLORS[coin], marker_line_width=0,
                hovertemplate=f"<b>{coin}</b>: %{{y:.2f}}%<extra></extra>",
            ))
        fig.update_layout(barmode="stack")
    else:
        datos  = get_rentabilidad_periodica(cripto, periodo)
        labels = [d["label"]   for d in datos]
        rets   = [d["retorno"] for d in datos]
        colors = ["#10b981" if r >= 2 else ("#f59e0b" if r >= 0 else "#ef4444") for r in rets]
        fig.add_trace(go.Bar(
            x=labels, y=rets, marker_color=colors, marker_line_width=0,
            text=[f"{'+' if r>=0 else ''}{r:.1f}%" for r in rets],
            textposition="outside", textfont=dict(size=9, color="#94a3b8"),
            hovertemplate="<b>%{x}</b><br>%{y:.2f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=28, b=10),
        font=dict(color="#94a3b8", size=10),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#64748b", size=9)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False,
                   tickfont=dict(color="#64748b", size=9), ticksuffix="%"),
        hoverlabel=dict(bgcolor="#0a1020", bordercolor="#3b82f6", font=dict(color="white")),
        bargap=0.3,
    )
    return fig
@callback(
    Output("pnl-summary-panel", "children"),
    Input("store-cripto", "data"),
    Input("dd-rango",     "value"),
)
def actualizar_pnl_panel(cripto, periodo):
    info   = get_rentabilidad_absoluta(cripto, periodo)
    p_pct  = info["pct"]
    p_abs  = info["abs"]
    t_pct  = info["total_pct"]
    t_abs  = info["total_abs"]
    inv    = info["inversion"]
    label  = PERIODO_LABEL.get(periodo, "")
    p_cls  = "pnl-val green" if p_pct >= 0 else "pnl-val red"
    t_cls  = "pnl-val green" if t_pct >= 0 else "pnl-val red"
    p_sign = "+" if p_pct >= 0 else ""
    t_sign = "+" if t_pct >= 0 else ""
    return [
        html.Div([
            html.Div(f"P&L {label}", className="pnl-label"),
            html.Div(f"{p_sign}{p_pct:.2f}%", className=p_cls),
            html.Div(f"{p_sign}${p_abs:,.2f}", className="pnl-sub"),
        ], className="pnl-stat-card"),
        html.Div([
            html.Div("Rentabilidad total", className="pnl-label"),
            html.Div(f"{t_sign}{t_pct:.1f}%", className=t_cls),
            html.Div(f"{t_sign}${t_abs:,.2f}", className="pnl-sub"),
        ], className="pnl-stat-card pnl-stat-card--total"),
        html.Div([
            html.Div("Capital invertido", className="pnl-label"),
            html.Div(f"${inv:,.0f}", className="pnl-val"),
            html.Div(f"{cripto}", className="pnl-sub"),
        ], className="pnl-stat-card"),
    ]
@callback(
    Output("tabla-operaciones", "children"),
    Input("store-cripto",       "data"),
    Input("intervalo-auto",     "n_intervals"),
)
def render_tabla_ops(cripto, _):
    ops = get_historial_operaciones()
    header = html.Div([
        html.Span("Fecha",  className="ops-th"),
        html.Span("Tipo",   className="ops-th"),
        html.Span("Cripto", className="ops-th"),
        html.Span("Precio", className="ops-th ops-right"),
        html.Span("P&L",    className="ops-th ops-right"),
    ], className="ops-row ops-header")
    rows = [header]
    for op in ops:
        tipo_cls = {"COMPRAR": "badge-green", "VENDER": "badge-red", "HOLD": "badge-amber"}.get(op["tipo"], "")
        pnl_cls  = "val-green" if op["pnl"] >= 0 else "val-red"
        sign     = "+" if op["pnl"] >= 0 else ""
        rows.append(html.Div([
            html.Span(op["fecha"],                className="ops-td ops-date"),
            html.Span(op["tipo"],                 className=f"ops-badge {tipo_cls}"),
            html.Span(op["cripto"],               className="ops-td ops-coin"),
            html.Span(_fmt(op["precio"]),         className="ops-td ops-right"),
            html.Span(f"{sign}{op['pnl']:.2f}%", className=f"ops-td ops-right {pnl_cls}"),
        ], className="ops-row"))
    return rows
# Modal open/close
@callback(
    Output("modal-wrapper", "style"),
    Input("btn-add-op",      "n_clicks"),
    Input("modal-cancel",    "n_clicks"),
    Input("modal-confirm",   "n_clicks"),
    Input("modal-close-x",   "n_clicks"),
    Input("modal-backdrop",  "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(*_):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered == "btn-add-op":
        return {"display": "flex", "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%", "zIndex": "1000", "alignItems": "center", "justifyContent": "center"}
    return {"display": "none"}
# Auto-fill price from current mock price when crypto changes
@callback(
    Output("modal-precio",     "value"),
    Output("modal-price-hint", "children"),
    Input("modal-cripto",      "value"),
)
def autofill_precio(cripto):
    if not cripto:
        return None, ""
    preds  = get_predicciones_lstm(cripto)
    precio = round(preds.get("precio_actual", BASE_PRICES.get(cripto, 0)), 2)
    hint   = f"Precio de mercado actual ({cripto}): {_fmt(precio)}"
    return precio, hint