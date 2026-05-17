import dash
from dash import html, dcc, callback, Input, Output, State
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_papertrade import (
    get_saldo_cash,
    get_criptomonedas,
    get_posiciones,
    get_historial,
    ejecutar_compra,
    ejecutar_venta,
)
from pages.api_client import get_predicciones_lstm_real
from pages.currency_utils import format_price, CURRENCY_RATES

dash.register_page(__name__, path="/papertrade", name="Paper Trade")

CAPITAL_INICIAL = 100_000.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _precio_actual(symbol: str) -> float:
    try:
        p = get_predicciones_lstm_real(symbol)
        return float(p.get("precio_actual") or 0)
    except Exception:
        return 0.0


def _pnl_color(valor: float) -> str:
    if valor > 0:
        return "#10b981"
    if valor < 0:
        return "#ef4444"
    return "#94a3b8"


def _pnl_str(pct: float) -> str:
    arrow = "▲" if pct >= 0 else "▼"
    return f"{arrow} {abs(pct):.2f}%"


# ── Layout ────────────────────────────────────────────────────────────────────

layout = html.Div([

    # Intervalo de refresco de precios (cada 2 min)
    dcc.Interval(id="pt-interval", interval=120_000, n_intervals=0),

    # ── Cabecera ──────────────────────────────────────────────────────────────
    html.Div([
        html.Span("Paper Trading", className="section-title", style={"fontSize": "1.5rem"}),
        html.Span("Simulador con 100 000 USD virtuales", style={"color": "#64748b", "fontSize": "13px", "marginLeft": "12px"}),
    ], style={"marginBottom": "24px"}),

    # ── Resumen de portfolio (4 KPIs) ─────────────────────────────────────────
    html.Div(id="pt-resumen", style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"}),

    # ── Fila principal ────────────────────────────────────────────────────────
    html.Div([

        # Columna izquierda: formulario de operación
        html.Div([
            html.Span("Nueva operación", className="section-title", style={"fontSize": "1.1rem"}),

            html.Div([
                # Selector cripto
                html.Div([
                    html.Label("Criptomoneda", style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="pt-select-crypto",
                        options=[],          # se rellena en callback
                        placeholder="Selecciona...",
                        clearable=False,
                        style={"background": "#0c1428", "color": "white", "border": "1px solid rgba(59,130,246,0.3)", "borderRadius": "8px"},
                    ),
                ], style={"marginBottom": "16px"}),

                # Tipo operación
                html.Div([
                    html.Label("Tipo", style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "6px"}),
                    dcc.RadioItems(
                        id="pt-tipo",
                        options=[
                            {"label": "  Comprar", "value": "BUY"},
                            {"label": "  Vender",  "value": "SELL"},
                        ],
                        value="BUY",
                        inline=True,
                        style={"color": "white", "gap": "20px"},
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"marginRight": "24px", "cursor": "pointer"},
                    ),
                ], style={"marginBottom": "16px"}),

                # Cantidad
                html.Div([
                    html.Label("Cantidad (unidades de cripto)", style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "6px"}),
                        dcc.Input(
                            id="pt-cantidad",
                            type="number",
                            min=0.01,
                            step=0.01,
                            value=0.01,
                            placeholder="Ej: 0.05",
                            style={
                                "width": "100%", "background": "#0c1428", "color": "white",
                                "border": "1px solid rgba(59,130,246,0.3)", "borderRadius": "8px",
                                "padding": "10px 14px", "fontSize": "15px",
                            },
                        ),
                ], style={"marginBottom": "16px"}),

                # Precio actual (informativo)
                html.Div(id="pt-precio-info", style={
                    "background": "rgba(59,130,246,0.08)", "borderRadius": "8px",
                    "padding": "10px 14px", "marginBottom": "16px",
                    "color": "#94a3b8", "fontSize": "13px",
                }),

                # Coste/ingreso estimado
                html.Div(id="pt-estimado", style={
                    "background": "rgba(16,185,129,0.08)", "borderRadius": "8px",
                    "padding": "10px 14px", "marginBottom": "20px",
                    "fontSize": "14px", "fontWeight": "600",
                }),

                # Botón ejecutar
                html.Button(
                    "Ejecutar operación",
                    id="pt-btn-ejecutar",
                    n_clicks=0,
                    style={
                        "width": "100%", "padding": "14px", "borderRadius": "10px",
                        "background": "linear-gradient(135deg, #2563eb, #1d4ed8)",
                        "color": "white", "border": "none", "fontWeight": "700",
                        "fontSize": "15px", "cursor": "pointer", "letterSpacing": "0.03em",
                    },
                ),

                # Feedback
                html.Div(id="pt-feedback", style={"marginTop": "14px", "fontSize": "13px", "minHeight": "20px"}),

            ], style={"marginTop": "20px"}),

        ], className="panel-ops-table", style={"flex": "1", "padding": "24px", "minWidth": "280px"}),

        # Columna derecha: posiciones + historial
        html.Div([

            # Posiciones abiertas
            html.Div([
                html.Span("Posiciones abiertas", className="section-title", style={"fontSize": "1.1rem"}),
                html.Div(id="pt-posiciones", style={"marginTop": "16px"}),
            ], className="panel-bar-chart", style={"padding": "24px", "marginBottom": "20px"}),

            # Historial de operaciones
            html.Div([
                html.Span("Historial de operaciones", className="section-title", style={"fontSize": "1.1rem"}),
                html.Div(id="pt-historial", style={"marginTop": "16px"}),
            ], className="panel-bar-chart", style={"padding": "24px"}),

        ], style={"flex": "2", "display": "flex", "flexDirection": "column", "gap": "0px"}),

    ], style={"display": "flex", "gap": "20px", "alignItems": "flex-start"}),

], className="page-container")


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("pt-select-crypto", "options"),
    Input("pt-interval", "n_intervals"),
)
def cargar_opciones_crypto(_):
    """Rellena el dropdown con las cryptos de la BD."""
    cryptos = get_criptomonedas()
    return [{"label": f"{c['nombre']} ({c['symbol']})", "value": c["symbol"]} for c in cryptos]


@callback(
    Output("pt-precio-info", "children"),
    Output("pt-estimado",    "children"),
    Input("pt-select-crypto", "value"),
    Input("pt-cantidad",      "value"),
    Input("pt-tipo",          "value"),
    Input("pt-interval",      "n_intervals"),
    State("store-currency",   "data"),
)
def actualizar_precio_estimado(symbol, cantidad, tipo, _, currency):
    if not symbol:
        return "Selecciona una criptomoneda para ver el precio.", ""

    precio = _precio_actual(symbol)
    if not precio:
        return f"Precio de {symbol} no disponible en este momento.", ""

    currency  = currency or "USD"
    rate      = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["rate"]
    sym_cur   = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]
    precio_fx = precio * rate

    precio_txt = f"Precio actual {symbol}: {sym_cur}{precio_fx:,.2f}"

    if not cantidad or float(cantidad) <= 0:
        return precio_txt, "Introduce una cantidad para ver el estimado."

    total     = float(cantidad) * precio * rate
    accion    = "Coste estimado" if tipo == "BUY" else "Ingreso estimado"
    color     = "#10b981" if tipo == "BUY" else "#f59e0b"
    estimado  = html.Span(
        f"{accion}: {sym_cur}{total:,.2f}",
        style={"color": color}
    )
    return precio_txt, estimado


@callback(
    Output("pt-feedback",  "children"),
    Output("pt-resumen",   "children"),
    Output("pt-posiciones","children"),
    Output("pt-historial", "children"),
    Input("pt-btn-ejecutar", "n_clicks"),
    Input("pt-interval",     "n_intervals"),
    State("pt-select-crypto","value"),
    State("pt-cantidad",     "value"),
    State("pt-tipo",         "value"),
    State("auth-token",      "data"),
    State("store-currency",  "data"),
    prevent_initial_call=False,
)
def ejecutar_y_refrescar(n_clicks, _, symbol, cantidad, tipo, username, currency):
    ctx      = dash.callback_context
    feedback = ""
    currency = currency or "USD"
    rate     = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["rate"]
    sym_cur  = CURRENCY_RATES.get(currency, CURRENCY_RATES["USD"])["symbol"]
    
    # ── Ejecutar operación si viene del botón ─────────────────────────────────
    if ctx.triggered and "pt-btn-ejecutar" in ctx.triggered[0]["prop_id"]:
        if not username:
            feedback = html.Span("⚠️ No has iniciado sesión.", style={"color": "#f59e0b"})
        elif not symbol:
            feedback = html.Span("⚠️ Selecciona una criptomoneda.", style={"color": "#f59e0b"})
        elif not cantidad or float(cantidad) <= 0:
            feedback = html.Span("⚠️ Introduce una cantidad válida.", style={"color": "#f59e0b"})
        else:
            precio = _precio_actual(symbol)
            if not precio:
                feedback = html.Span("⚠️ No se pudo obtener el precio. Intenta de nuevo.", style={"color": "#f59e0b"})
            else:
                fn  = ejecutar_compra if tipo == "BUY" else ejecutar_venta
                ok, msg = fn(username, symbol, float(cantidad), precio)
                color    = "#10b981" if ok else "#ef4444"
                feedback = html.Span(msg, style={"color": color})

    # ── Resumen de portfolio ──────────────────────────────────────────────────
    if not username:
        resumen_ui  = html.Span("Inicia sesión para ver tu portfolio.", style={"color": "#64748b"})
        posiciones_ui = html.Span("—")
        historial_ui  = html.Span("—")
        return feedback, resumen_ui, posiciones_ui, historial_ui

    cash       = get_saldo_cash(username)
    posiciones = get_posiciones(username)

    # Valor de mercado de posiciones con precio actual
    valor_mercado = 0.0
    coste_total   = 0.0
    for pos in posiciones:
        precio_ahora  = _precio_actual(pos["symbol"])
        valor_mercado += precio_ahora * pos["qty_net"]
        coste_total   += pos["precio_medio"] * pos["qty_net"]

    patrimonio_total = cash + valor_mercado
    pnl_abs          = valor_mercado - coste_total
    pnl_pct          = (pnl_abs / coste_total * 100) if coste_total > 0 else 0.0
    rentabilidad_total_pct = ((patrimonio_total - CAPITAL_INICIAL) / CAPITAL_INICIAL) * 100

    def _kpi(label, valor_str, color="#e2e8f0", hint=None):
        return html.Div([
            html.Div(label, style={"fontSize": "11px", "color": "#64748b", "marginBottom": "4px", "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em"}),
            html.Div(valor_str, style={"fontSize": "1.4rem", "fontWeight": "800", "color": color}),
            html.Div(hint, style={"fontSize": "11px", "color": "#475569", "marginTop": "4px"}) if hint else None,
        ], className="pnl-stat-card", style={"flex": "1", "minWidth": "160px", "padding": "18px 20px"})

    resumen_ui = [
        _kpi("Patrimonio total",     f"{sym_cur}{patrimonio_total * rate:,.2f}", "#e2e8f0", "cash + posiciones"),
        _kpi("Cash disponible",      f"{sym_cur}{cash * rate:,.2f}", "#3b82f6"),
        _kpi("PnL posiciones",       _pnl_str(pnl_pct), _pnl_color(pnl_pct), f"{sym_cur}{pnl_abs * rate:,.2f}"),
        _kpi("Rentabilidad total",   _pnl_str(rentabilidad_total_pct), _pnl_color(rentabilidad_total_pct), "vs 100 000 inicial"),
    ]

    # ── Posiciones abiertas ───────────────────────────────────────────────────
    if not posiciones:
        posiciones_ui = html.Div("No tienes posiciones abiertas.", style={"color": "#64748b", "fontSize": "14px"})
    else:
        filas = []
        for pos in posiciones:
            precio_ahora = _precio_actual(pos["symbol"])
            val_ahora    = precio_ahora * pos["qty_net"]
            coste        = pos["precio_medio"] * pos["qty_net"]
            pnl_p        = ((val_ahora - coste) / coste * 100) if coste > 0 else 0.0
            pnl_a        = val_ahora - coste

            filas.append(html.Div([
                # Symbol + nombre
                html.Div([
                    html.Span(pos["symbol"], style={"fontWeight": "800", "color": "white", "fontSize": "15px"}),
                    html.Span(pos["nombre"], style={"color": "#64748b", "fontSize": "11px", "marginLeft": "8px"}),
                ], style={"flex": "2"}),
                # Cantidad neta
                html.Div([
                    html.Div("Cantidad", style={"fontSize": "10px", "color": "#64748b"}),
                    html.Div(f"{pos['qty_net']:.3f}", style={"fontSize": "13px", "color": "#e2e8f0"}),
                ], style={"flex": "1", "textAlign": "right"}),
                # Precio medio compra
                html.Div([
                    html.Div("P. medio", style={"fontSize": "10px", "color": "#64748b"}),
                    html.Div(f"{sym_cur}{pos['precio_medio'] * rate:,.2f}", style={"fontSize": "13px", "color": "#e2e8f0"}),
                ], style={"flex": "1", "textAlign": "right"}),
                # Precio actual
                html.Div([
                    html.Div("P. actual", style={"fontSize": "10px", "color": "#64748b"}),
                    html.Div(f"{sym_cur}{precio_ahora * rate:,.2f}", style={"fontSize": "13px", "color": "#e2e8f0"}),
                ], style={"flex": "1", "textAlign": "right"}),
                # PnL %
                html.Div([
                    html.Div("PnL", style={"fontSize": "10px", "color": "#64748b"}),
                    html.Div(_pnl_str(pnl_p), style={"fontSize": "14px", "fontWeight": "700", "color": _pnl_color(pnl_p)}),
                    html.Div(f"{sym_cur}{pnl_a * rate:,.2f}", style={"fontSize": "10px", "color": _pnl_color(pnl_a)}),
                ], style={"flex": "1", "textAlign": "right"}),

            ], style={
                "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                "padding": "14px 16px", "borderRadius": "10px", "marginBottom": "10px",
                "background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.05)",
            }))

        posiciones_ui = html.Div(filas)

    # ── Historial ─────────────────────────────────────────────────────────────
    historial = get_historial(username, limit=30)
    if not historial:
        historial_ui = html.Div("Aún no has realizado operaciones.", style={"color": "#64748b", "fontSize": "14px"})
    else:
        cabecera = html.Div([
            html.Div("Tipo",    style={"flex": "1", "fontSize": "11px", "color": "#64748b", "fontWeight": "700"}),
            html.Div("Cripto",  style={"flex": "1", "fontSize": "11px", "color": "#64748b", "fontWeight": "700"}),
            html.Div("Cantidad",style={"flex": "1", "fontSize": "11px", "color": "#64748b", "fontWeight": "700", "textAlign": "right"}),
            html.Div("Precio",  style={"flex": "1", "fontSize": "11px", "color": "#64748b", "fontWeight": "700", "textAlign": "right"}),
            html.Div("Total",   style={"flex": "1", "fontSize": "11px", "color": "#64748b", "fontWeight": "700", "textAlign": "right"}),
            html.Div("Fecha",   style={"flex": "2", "fontSize": "11px", "color": "#64748b", "fontWeight": "700", "textAlign": "right"}),
        ], style={"display": "flex", "padding": "6px 16px", "marginBottom": "6px"})

        filas_h = [cabecera]
        for op in historial:
            es_buy  = op["tipo"] == "BUY"
            color_t = "#10b981" if es_buy else "#ef4444"
            filas_h.append(html.Div([
                html.Div(
                    html.Span(op["tipo"], style={"background": f"{'rgba(16,185,129,0.15)' if es_buy else 'rgba(239,68,68,0.15)'}", "color": color_t, "padding": "2px 10px", "borderRadius": "20px", "fontSize": "11px", "fontWeight": "700"}),
                    style={"flex": "1"}
                ),
                html.Div(op["symbol"], style={"flex": "1", "color": "#e2e8f0", "fontWeight": "600"}),
                html.Div(f"{op['cantidad']:.6f}",                style={"flex": "1", "color": "#94a3b8", "textAlign": "right", "fontSize": "12px"}),
                html.Div(f"{sym_cur}{op['precio'] * rate:,.2f}", style={"flex": "1", "color": "#94a3b8", "textAlign": "right", "fontSize": "12px"}),
                html.Div(f"{sym_cur}{op['valor_total'] * rate:,.2f}", style={"flex": "1", "color": color_t, "textAlign": "right", "fontSize": "12px", "fontWeight": "600"}),
                html.Div(str(op["fecha"])[:16],                  style={"flex": "2", "color": "#475569", "textAlign": "right", "fontSize": "11px"}),
            ], style={
                "display": "flex", "alignItems": "center",
                "padding": "10px 16px", "borderRadius": "8px", "marginBottom": "6px",
                "background": "rgba(255,255,255,0.02)", "border": "1px solid rgba(255,255,255,0.04)",
            }))

        historial_ui = html.Div(filas_h)

    return feedback, resumen_ui, posiciones_ui, historial_ui