import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pages.mock_data import (
    get_historico,
    get_rentabilidad_periodica, get_rentabilidad_all,
    get_rentabilidad_absoluta,
    CRYPTO_COLORS
)

from pages.currency_utils import format_price, CURRENCY_RATES
from pages.api_client import get_predicciones_lstm_real, get_decision_rl

from database.db_utils import get_trades, get_last_operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

dash.register_page(__name__, path="/home/", name="Visión General")
CRIPTOS    = ["ALL", "BTC", "ETH", "SOL", "AVAX"]
CARD_COINS = ["BTC", "ETH", "SOL"]
PERIODO_LABEL = {"1d": "Diaria", "7d": "Semanal", "1m": "Mensual"}
# Se usa format_price importado de lstm_utils
def _mini_chart(coin):
    df = get_historico(coin, 24)
    if df.empty or len(df) < 2:
        return go.Figure()
    close = df["close"].tolist()
    preds = get_predicciones_lstm_real(coin)
    p = preds.get("precio_actual", 0)
    # Usar precio actual como último punto si es válido
    if p and abs(p - close[-1]) > 1e-6:
        close = close[:-1] + [p]
    # Calcular variación diaria (badge)
    c = ((close[-1] - close[0]) / close[0]) * 100 if close[0] != 0 else 0
    color = "#10b981" if c >= 0 else "#ef4444"
    fill  = "rgba(16,185,129,0.12)" if c >= 0 else "rgba(239,68,68,0.10)"
    fig = go.Figure(go.Scatter(x=list(range(len(close))), y=close, fill="tozeroy", fillcolor=fill,
                               line=dict(color=color, width=2.0, shape="spline"), hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig
def chart_card_skeleton(coin):
    """Genera la estructura estática de la card sin datos."""
    return html.Div([
        html.Div([
            html.Div([
                html.Span(coin, className="card-coin-name"),
                html.Span(id=f"card-badge-{coin}", className="card-badge"),
            ], className="card-coin-row"),
            html.Button("⛶", className="btn-expand", title="Ampliar"),
        ], className="chart-card-header"),
        dcc.Loading(
            html.Div(id=f"card-price-{coin}", className="card-price"),
            type="default", color="#3b82f6"
        ),
        dcc.Loading(
            dcc.Graph(id=f"mini-chart-{coin}", config={"displayModeBar": False, "staticPlot": True}, className="mini-chart"),
            type="default", color="#3b82f6"
        ),
    ], className="chart-card")
# ── LAYOUT ────────────────────────────────────────────────────────────────────
layout = html.Div([
    # Selector ALL | BTC | ETH | SOL | AVAX
    html.Div([
        *[html.Button(c, id={"type": "btn-cripto", "index": c},
            className="btn-cripto" + (" btn-cripto-active" if c == "ALL" else ""))
          for c in CRIPTOS],
        html.Div(id="timestamp-header", className="page-timestamp"),
    ], className="cripto-selector"),
    # 3 Chart Cards Estáticas
    html.Div([
        dcc.Link(chart_card_skeleton(c), href=f"/predictions?coin={c}", style={"textDecoration": "none"})
        for c in CARD_COINS
    ], className="chart-cards-row"),
    # Bottom: Bar Chart + Table
    html.Div([
        # Left – P&L summary + bar chart
        html.Div([
            # Absolute P&L panel
            html.Div(id="pnl-summary-panel", className="pnl-summary-panel", style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "15px", "marginBottom": "25px"}),
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
                          config={"displayModeBar": True, "scrollZoom": True, "modeBarButtonsToAdd": ['zoomIn2d', 'zoomOut2d']}, style={"height": "240px"}),
            ], className="bar-chart-inner"),
        ], className="panel-bar-chart"),
        # Right – Operations table
        html.Div([
            html.Div([
                html.Span("Acciones realizadas", className="section-title"),
                html.Button(id="btn-add-op", className="btn-add-op",
                            n_clicks=0, title="Añadir operación manual", style={"display": "none"}),
            ], className="section-header"),
            html.Div(id="tabla-operaciones", className="ops-table-wrapper"),
        ], className="panel-ops-table"),
    ], className="bottom-row"),
], className="page-container")
# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(
    Output("overview-greeting", "children"),
    Input("auth-token", "data"),
)
def actualizar_saludo(auth):
    from datetime import datetime
    hora = datetime.now().hour
    if hora < 12:
        saludo = "Buenos días"
    elif hora < 20:
        saludo = "Buenas tardes"
    else:
        saludo = "Buenas noches"
    nombre = auth if auth else "Trader"
    return f"{saludo}, {nombre} 👋"

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
    preds = get_predicciones_lstm_real(c)
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
    Input("store-currency", "data"),
)
def actualizar_pnl_panel(cripto, periodo, currency):
    if not currency: currency = "USD"
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
            html.Div(f"{p_sign}{format_price(p_abs, currency)}", className="pnl-sub"),
        ], className="pnl-stat-card"),
        html.Div([
            html.Div("Rentabilidad total", className="pnl-label"),
            html.Div(f"{t_sign}{t_pct:.1f}%", className=t_cls),
            html.Div(f"{t_sign}{format_price(t_abs, currency)}", className="pnl-sub"),
        ], className="pnl-stat-card pnl-stat-card--total"),
        html.Div([
            html.Div("Capital invertido", className="pnl-label"),
            html.Div(format_price(inv, currency), className="pnl-val"),
            html.Div(f"{cripto}", className="pnl-sub"),
        ], className="pnl-stat-card"),
    ]
@callback(
    Output("tabla-operaciones", "children"),
    Input("store-cripto",       "data"),
    Input("store-currency",     "data"),
    Input("intervalo-auto",     "n_intervals"),
)
def render_tabla_ops(cripto, currency, _):
    if not currency: currency = "USD"
    ops = get_trades()
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
            html.Span(format_price(op["precio"], currency), className="ops-td ops-right"),
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
    preds  = get_predicciones_lstm_real(cripto)
    precio = round(preds.get("precio_actual", 0), 2)
    hint   = f"Precio de mercado actual ({cripto}): {format_price(precio)}"
    return precio, hint

@callback(
    [Output(f"card-price-{c}", "children") for c in CARD_COINS] +
    [Output(f"card-badge-{c}", "children") for c in CARD_COINS] +
    [Output(f"card-badge-{c}", "className") for c in CARD_COINS] +
    [Output(f"mini-chart-{c}", "figure") for c in CARD_COINS],
    Input("store-currency", "data"),
    Input("intervalo-auto",  "n_intervals"),
)
def update_all_cards(currency, _):
    if not currency: currency = "USD"
    prices, badges, b_classes, figs = [], [], [], []
    
    for coin in CARD_COINS:
        preds = get_predicciones_lstm_real(coin)
        p = preds.get("precio_actual", 0)
        
        df = get_historico(coin, 1) # Last 24 hours
        if not df.empty and len(df) >= 24 and p > 0:
            c = ((p - df.iloc[0]["close"]) / df.iloc[0]["close"]) * 100
        else:
            c = 0
        
        prices.append(format_price(p, currency))
        badges.append(f"{'+' if c>=0 else ''}{c:.2f}%")
        b_classes.append(f"card-badge {'badge-green' if c>=0 else 'badge-red'}")
        figs.append(_mini_chart(coin))
        
    return prices + badges + b_classes + figs