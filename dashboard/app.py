import os
import sys
import requests
import json
# Solución para el error OMP #15 en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Añadir el directorio raíz al path para que las importaciones de 'core' funcionen
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(root_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import dash
import logging
from dash import Dash, html, dcc, Input, Output, State, callback
from core.cache_config import setup_cache, cache
# El procesamiento local se ha movido a la nube (ngrok)
from pages.api_client import _prepare_X_input, preload_all_data



# Configuración de logging básica
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import diskcache
cache_obj = diskcache.Cache("./cache_dir")

# Inicializar precarga de datos
preload_all_data()

app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    suppress_callback_exceptions=True,
    title="NeuroQuant · Crypto AI",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    background_callback_manager=dash.DiskcacheManager(cache_obj)
)

cache = setup_cache(app)

# Configuración de la aplicación
logger.info("📡 Dashboard configurado para modo Cloud (ngrok)")


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
        html.Div(id="topbar-user-name", style={"fontWeight": "600", "color": "var(--text-primary)", "fontSize": "13px", "marginRight": "12px"}),
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
        dcc.Link(html.Div(["◈ ", html.Span("Visión General")],  className="sidebar-item"), href="/home"),
        dcc.Link(html.Div(["📈 ", html.Span("Paper Trading")], className="sidebar-item"), href="/papertrade"),
    ], className="sidebar-section"),
    # Section 2
    html.Div([
        html.Div("💹  MERCADOS", className="sidebar-section-label"),
        dcc.Link(html.Div(["₿ ", html.Span("Bitcoin  (BTC)")],  className="sidebar-item"), href="/predictions?coin=BTC"),
        dcc.Link(html.Div(["Ξ ", html.Span("Ethereum (ETH)")],  className="sidebar-item"), href="/predictions?coin=ETH"),
        dcc.Link(html.Div(["◎ ", html.Span("Solana   (SOL)")],  className="sidebar-item"), href="/predictions?coin=SOL"),
        dcc.Link(html.Div(["⟠ ", html.Span("Avalanche (AVAX)")],  className="sidebar-item"), href="/predictions?coin=AVAX"),
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
    dcc.Location(id="global-url", refresh=True),
    dcc.Store(id="store-cripto",       data="BTC"),
    dcc.Store(id="store-predicciones", data={}),
    dcc.Store(id="store-decision",     data={}),
    dcc.Store(id="store-currency",     data="USD", storage_type="local"),
    dcc.Store(id="auth-token",         storage_type="session"),
    dcc.Interval(id="intervalo-auto",  interval=5 * 60 * 1000, n_intervals=0),
    html.Div(topbar, id="topbar-container"),
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


@callback(
    Output("global-url", "pathname", allow_duplicate=True),
    Input("global-url", "pathname"),
    State("auth-token", "data"),
    prevent_initial_call=True
)
def redirect_root_and_login(pathname, auth):
    if not auth and pathname == "/":
        return "/login"
    if auth and pathname == "/login":
        return "/home"
    if not auth and pathname == "/home":
        return "/login"
    return dash.no_update


@callback(
    Output("topbar-user-name", "children"),
    Input("auth-token", "data")
)
def update_topbar_user(auth):
    return f"Hola, {auth}" if auth else ""



if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)