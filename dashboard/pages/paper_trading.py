import dash
from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
from database.db_utils import get_trades
from pages.currency_utils import format_price

dash.register_page(__name__, path="/paper-trading", name="Paper Trading")

layout = html.Div([
    html.Div([
        html.Div("📈 Paper Trading", className="page-hero-title"),
        html.Div("Simulación de bolsa y operaciones con dinero digital.", className="page-hero-sub")
    ], className="page-hero"),
    
    html.Div([
        html.Div([
            html.Div("Balance de la Cuenta", style={"color": "var(--text-secondary)", "marginBottom": "10px", "fontSize": "14px", "fontWeight": "600", "textTransform": "uppercase"}),
            html.Div(id="pt-balance-display", style={"fontSize": "32px", "fontWeight": "bold", "color": "var(--green)"}),
        ], className="panel-card", style={"marginBottom": "20px"}),
        
        html.Div([
            html.Div("Nueva Operación", className="section-title", style={"marginBottom": "15px"}),
            html.Div([
                html.Label("Criptomoneda", className="modal-label"),
                dcc.Dropdown(id="pt-cripto", options=[{"label": c, "value": c} for c in ["BTC", "ETH", "SOL", "AVAX"]], value="BTC", className="modal-dropdown", clearable=False),
            ], className="modal-field"),
            html.Div([
                html.Label("Tipo de Orden", className="modal-label"),
                dcc.Dropdown(id="pt-tipo", options=[{"label": "COMPRAR", "value": "COMPRAR"}, {"label": "VENDER", "value": "VENDER"}], value="COMPRAR", className="modal-dropdown", clearable=False),
            ], className="modal-field"),
            html.Div([
                html.Label("Cantidad", className="modal-label"),
                dcc.Input(id="pt-cantidad", type="number", value=0.1, step=0.01, className="modal-input"),
            ], className="modal-field"),
            html.Div(id="pt-precio-actual", style={"marginTop": "15px", "marginBottom": "15px", "color": "var(--text-secondary)", "fontFamily": "'JetBrains Mono', monospace", "fontSize": "13px"}),
            html.Button("Ejecutar Operación", id="pt-btn-ejecutar", className="btn-modal-confirm", n_clicks=0, style={"width": "100%", "marginTop": "10px", "padding": "16px", "fontSize": "16px", 
                "fontWeight": "bold", "backgroundColor": "var(--blue)", "color": "white", 
                "border": "none", "borderRadius": "12px", "cursor": "pointer", 
                "boxShadow": "0 4px 15px rgba(59, 130, 246, 0.3)"}),
            html.Div(id="pt-mensaje", style={"marginTop": "15px", "fontWeight": "bold", "fontSize": "13px"})
        ], className="panel-card", style={"marginBottom": "20px"}),
        
        html.Div([
            html.Div([
                html.Span("Historial de Operaciones", className="section-title"),
            ], className="section-header"),
            html.Div(id="pt-tabla-operaciones", className="ops-table-wrapper")
        ], className="panel-ops-table")
    ], style={"maxWidth": "800px", "margin": "0 auto"})
], className="page-container")

@callback(
    Output("pt-balance-display", "children"),
    Input("pt-btn-ejecutar", "n_clicks"),
    Input("store-currency", "data")
)
def update_balance_display(n, currency):
    if not currency: currency = "USD"
    bal = get_balance()
    return format_price(bal, currency)

@callback(
    Output("pt-precio-actual", "children"),
    Input("pt-cripto", "value"),
    Input("store-currency", "data")
)
def update_current_price(cripto, currency):
    if not currency: currency = "USD"
    if not cripto: return ""
    preds = get_predicciones_lstm_real(cripto)
    precio = preds.get("predicted_price", 0)
    return f"Precio Actual: {format_price(precio, currency)}"

@callback(
    Output("pt-mensaje", "children"),
    Output("pt-mensaje", "style"),
    Input("pt-btn-ejecutar", "n_clicks"),
    State("pt-cripto", "value"),
    State("pt-tipo", "value"),
    State("pt-cantidad", "value"),
    prevent_initial_call=True
)
def ejecutar_operacion(n, cripto, tipo, cantidad):
    if n == 0 or not cantidad or cantidad <= 0:
        return dash.no_update, dash.no_update
    preds = get_predicciones_lstm_real(cripto)
    precio = preds.get("precio_actual", 0)
    if precio <= 0:
        return "Error al obtener el precio actual", {"color": "var(--red)", "marginTop": "15px", "fontWeight": "bold"}
    
    success, msg = add_trade(cripto, tipo, float(cantidad), precio)
    if success:
        return msg, {"color": "var(--green)", "marginTop": "15px", "fontWeight": "bold"}
    else:
        return msg, {"color": "var(--red)", "marginTop": "15px", "fontWeight": "bold"}

@callback(
    Output("pt-tabla-operaciones", "children"),
    Input("pt-btn-ejecutar", "n_clicks"),
    Input("store-currency", "data")
)
def render_pt_table(n, currency):
    if not currency: currency = "USD"
    ops = get_trades()
    header = html.Div([
        html.Span("Fecha",  className="ops-th"),
        html.Span("Tipo",   className="ops-th"),
        html.Span("Cripto", className="ops-th"),
        html.Span("Cantidad", className="ops-th"),
        html.Span("Precio", className="ops-th ops-right"),
        html.Span("Total",    className="ops-th ops-right"),
    ], className="ops-row ops-header", style={"gridTemplateColumns": "130px 70px 50px 70px 1fr 1fr"})
    rows = [header]
    for op in ops:
        tipo_cls = {"COMPRAR": "badge-green", "VENDER": "badge-red"}.get(op["tipo"], "")
        rows.append(html.Div([
            html.Span(op["fecha"],                className="ops-td ops-date"),
            html.Span(op["tipo"],                 className=f"ops-badge {tipo_cls}"),
            html.Span(op["cripto"],               className="ops-td ops-coin"),
            html.Span(f"{op['cantidad']:.4f}",    className="ops-td"),
            html.Span(format_price(op["precio"], currency), className="ops-td ops-right"),
            html.Span(format_price(op["total"], currency), className="ops-td ops-right"),
        ], className="ops-row", style={"gridTemplateColumns": "130px 70px 50px 70px 1fr 1fr"}))
    return rows
