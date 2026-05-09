import dash
import os
import sys
from dash import Dash, html, dcc, Input, Output, State, callback

# Añadir el directorio raíz al path para que las importaciones de 'dashboard' funcionen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard.pages.lstm_utils import get_predicciones_lstm_real
app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="NeuroQuant · Crypto AI",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
server = app.server
# ── TOPBAR ──────────────────────────────────────────────────────────────────
topbar = html.Div([
    html.Div([
        html.Button(
            html.Div([html.Span(), html.Span(), html.Span()], className="ham-lines"),
            id="btn-sidebar-toggle", className="hamburger-btn", n_clicks=0
        ),
        html.Div([
            html.Span("Neuro", className="logo-neuro"),
            html.Span("Quant", className="logo-quant"),
            html.Span(" AI", className="logo-ai"),
        ], className="logo-text"),
    ], className="topbar-left"),
    html.Div([
        dcc.Link(
            html.Button("⚙", id="btn-settings", className="icon-btn", title="Ajustes"),
            href="/settings"
        ),
        dcc.Link(
            html.Button("👤", id="btn-profile", className="icon-btn icon-btn-profile", title="Perfil"),
            href="/profile"
        ),
    ], className="topbar-right"),
], className="topbar")
# ── SIDEBAR BACKDROP ────────────────────────────────────────────────────────
backdrop = html.Div(id="sidebar-backdrop", className="sidebar-backdrop", n_clicks=0)
# ── SIDEBAR ─────────────────────────────────────────────────────────────────
sidebar = html.Div([
    # Section 1
    html.Div([
        html.Div("📊  ANÁLISIS", className="sidebar-section-label"),
        dcc.Link(html.Div(["◈ ", html.Span("Visión General")],  className="sidebar-item"), href="/"),
    ], className="sidebar-section"),
    # Section 2
    html.Div([
        html.Div("💹  MERCADOS", className="sidebar-section-label"),
        dcc.Link(html.Div(["₿ ", html.Span("Bitcoin  (BTC)")],  className="sidebar-item"), href="/predictions?coin=BTC"),
        dcc.Link(html.Div(["Ξ ", html.Span("Ethereum (ETH)")],  className="sidebar-item"), href="/predictions?coin=ETH"),
        dcc.Link(html.Div(["◎ ", html.Span("Solana   (SOL)")],  className="sidebar-item"), href="/predictions?coin=SOL"),
    ], className="sidebar-section"),
    # Section 3
    html.Div([
        html.Div("⚙  CUENTA", className="sidebar-section-label"),
        dcc.Link(html.Div(["👤 ", html.Span("Mi Perfil")], className="sidebar-item"), href="/profile"),
        dcc.Link(html.Div(["⚙ ", html.Span("Ajustes")],   className="sidebar-item"), href="/settings"),
    ], className="sidebar-section"),
    html.Button(
        ["← ", html.Span("Regresar")],
        id="btn-sidebar-close", className="sidebar-close-btn", n_clicks=0
    ),
], id="sidebar", className="sidebar")
# ── APP LAYOUT ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="global-url", refresh=False),
    dcc.Store(id="store-cripto",       data="BTC"),
    dcc.Store(id="store-predicciones", data={}),
    dcc.Store(id="store-decision",     data={}),
    dcc.Store(id="store-currency",     data="USD", storage_type="local"),
    dcc.Interval(id="intervalo-auto",  interval=5 * 60 * 1000, n_intervals=0),
    topbar,
    backdrop,
    sidebar,
    html.Div(dash.page_container, className="page-wrapper"),
], className="dashboard-root")
# ── SIDEBAR TOGGLE CALLBACK ──────────────────────────────────────────────────
@callback(
    Output("sidebar",          "className"),
    Output("sidebar-backdrop", "className"),
    Input("btn-sidebar-toggle", "n_clicks"),
    Input("btn-sidebar-close",  "n_clicks"),
    Input("sidebar-backdrop",   "n_clicks"),
    State("sidebar", "className"),
    prevent_initial_call=True,
)
def toggle_sidebar(open_n, close_n, back_n, current):
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    if triggered == "btn-sidebar-toggle":
        return "sidebar open", "sidebar-backdrop show"
    return "sidebar", "sidebar-backdrop"
@callback(
    Output("store-cripto", "data"),
    Input("global-url", "search"),
    Input({"type": "btn-cripto", "index": dash.ALL}, "n_clicks"),
    State("store-cripto", "data"),
)
def sync_cripto(search, btn_clicks, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    prop_id = ctx.triggered[0]["prop_id"]
    if "btn-cripto" in prop_id:
        val = ctx.triggered[0]["value"]
        if val:
            import json
            btn_id = json.loads(prop_id.split(".")[0])
            return btn_id["index"]
            
    if "global-url" in prop_id and search:
        import urllib.parse
        parsed = urllib.parse.parse_qs(search.lstrip('?'))
        if 'coin' in parsed:
            return parsed['coin'][0]
            
    return dash.no_update

@callback(
    Output("store-currency", "data"),
    Input("sett-currency", "value"),
    prevent_initial_call=True
)
def update_currency_store(currency):
    return currency if currency else "USD"
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        print("\n🚀 Iniciando pre-carga de modelos AI...")
        for c in ["BTC", "ETH", "SOL", "AVAX"]:
            get_predicciones_lstm_real(c)
        print("✅ Todos los modelos están listos.\n")
    
    app.run(debug=True)
