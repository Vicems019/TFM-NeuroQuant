import dash
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
from database.db_utils import get_trades, get_balance, DB_NAME
import os
from datetime import datetime

dash.register_page(__name__, path="/profile", name="Perfil")

# --- Modern Card-based Profile Layout ---
layout = html.Div([
    html.Div([
        html.Div([
            # Hero Card
            html.Div([
                html.Div("NQ", className="profile-avatar-wrap", style={"fontSize": "2.2rem", "fontWeight": "bold"}),
                html.Div([
                    html.Div(id="profile-display-name", className="profile-hero-name"),
                    html.Div("Trader · NeuroQuant AI", className="profile-hero-role"),
                    html.Div([
                        html.Span("●", style={"color": "#10b981", "fontSize": "1.1em", "marginRight": "6px"}),
                        html.Span("Cuenta activa", className="profile-status-text"),
                    ], className="profile-status-badge"),
                ], className="profile-hero-info"),
            ], className="profile-hero-card"),

            # KPIs
            html.Div([
                html.Div([
                    html.Div("📄", className="kpi-icon"),
                    html.Div(id="kpi-operaciones", className="kpi-val"),
                    html.Div("Operaciones", className="kpi-label"),
                ], className="kpi-card"),
                html.Div([
                    html.Div("💰", className="kpi-icon"),
                    html.Div(id="kpi-pnl", className="kpi-val"),
                    html.Div("P&L total", className="kpi-label"),
                ], className="kpi-card"),
                html.Div([
                    html.Div("🏆", className="kpi-icon"),
                    html.Div(id="kpi-winrate", className="kpi-val"),
                    html.Div("Win rate", className="kpi-label"),
                ], className="kpi-card"),
                html.Div([
                    html.Div("💵", className="kpi-icon"),
                    html.Div(id="profile-balance", className="kpi-val"),
                    html.Div("Balance (papertrade)", className="kpi-label"),
                    html.Div(id="profile-balance-upd", className="profile-kpi-mini-label", style={"marginTop": "2px"}),
                ], className="kpi-card"),
            ], className="profile-kpis-grid", style={"margin": "32px 0 18px 0"}),

            # Info Panels
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Datos de cuenta", className="panel-title", style={"marginBottom": "10px"}),
                        html.Div([
                            html.Div([
                                html.Span("Correo", className="info-label"),
                                html.Span("usuario@tfm.es", className="info-val"),
                            ], className="info-row"),
                            html.Div([
                                html.Span("Plan", className="info-label"),
                                html.Span("NeuroQuant Pro", className="info-val", style={"color": "#10b981"}),
                            ], className="info-row"),
                            html.Div([
                                html.Span("Activo desde", className="info-label"),
                                html.Span(id="profile-activo-desde", className="info-val"),
                            ], className="info-row"),
                        ], className="info-table"),
                    ], className="profile-panel", style={"background": "var(--bg-card2)", "boxShadow": "0 2px 12px rgba(59,130,246,0.08)", "marginBottom": "28px", "marginRight": "16px", "padding": "28px 32px 32px"}),
                ]),
                html.Div([
                    html.Div("Modelo e inferencia", className="panel-title", style={"marginBottom": "10px"}),
                    html.Div([
                        html.Div([
                            html.Span("Modelo activo", className="info-label"),
                            html.Span("LSTM + RL PPO v2.1", className="info-val"),
                        ], className="info-row"),
                        html.Div([
                            html.Span("Mercados", className="info-label"),
                            html.Span("BTC · ETH · SOL · AVAX", className="info-val"),
                        ], className="info-row"),
                        html.Div([
                            html.Span("Actualización", className="info-label"),
                            html.Span("Cada 5 minutos · última sync hace ~2 min", className="info-val"),
                        ], className="info-row"),
                    ], className="info-table"),
                ], className="profile-panel", style={"marginBottom": "28px"}),
            ], className="profile-panels-row", style={"margin": "28px 0 0 0", "gap": "32px"}),

            # Danger Zone (Red Card, larger, popup confirmation)
            html.Div([
                html.Div("Zona de riesgo", className="profile-danger-title", style={"fontSize": "1.1rem", "marginBottom": "10px"}),
                html.P(
                    "Eliminar la sesión en el panel. Si en el futuro hubiera cuenta persistente, aquí se daría de baja definitivamente.",
                    className="profile-danger-copy",
                ),
                html.Div([
                    html.Button(
                        "Eliminar cuenta",
                        id="btn-del-popup",
                        className="btn-account-delete",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "fontSize": "1.1rem",
                            "padding": "16px 0",
                            "marginTop": "10px",
                            "background": "linear-gradient(90deg, #ef4444, #b91c1c)",
                            "color": "#fff",
                            "border": "none",
                            "fontWeight": "700",
                            "fontSize": "15px",
                            "borderRadius": "10px",
                            "boxShadow": "0 2px 12px rgba(239,68,68,0.18)"
                        }
                    ),
                ], style={"marginTop": "18px"}),
                dcc.ConfirmDialog(
                    id="confirm-del-1",
                    message="¿Estás seguro de que quieres cerrar tu cuenta? Esta acción es irreversible.",
                ),
                dcc.ConfirmDialog(
                    id="confirm-del-2",
                    message="¿Seguro que quieres cerrar tu cuenta? Se cerrará tu sesión y se eliminarán tus datos de sesión.",
                ),
                html.Button("Cerrar sesión", id="btn-logout", className="btn-logout btn-logout-amber", n_clicks=0, style={"marginTop": "32px", "background": "linear-gradient(90deg, #f59e0b, #fbbf24)", "color": "#fff", "border": "none", "fontWeight": "700", "fontSize": "15px", "borderRadius": "10px", "padding": "14px 0", "width": "100%", "boxShadow": "0 2px 12px rgba(245,158,11,0.18)"}),
            ], className="profile-danger-zone", style={"marginTop": "40px", "background": "linear-gradient(120deg, #ef4444 0%, #b91c1c 100%)", "border": "1.5px solid #ef4444", "boxShadow": "0 2px 16px rgba(239,68,68,0.13)", "padding": "36px 32px"}),
        ], className="profile-content"),
    ], className="profile-page-inner"),
], className="page-container")


@callback(
    Output("profile-display-name", "children"),
    Output("kpi-operaciones", "children"),
    Output("kpi-pnl", "children"),
    Output("kpi-winrate", "children"),
    Output("profile-activo-desde", "children"),
    Output("profile-balance", "children"),
    Output("profile-balance-upd", "children"),
    Input("auth-token", "data"),
)
def populate_profile(auth):
    nombre = auth if auth else "Trader"
    ops = get_trades()
    total = len(ops)
    ganadas = sum(1 for o in ops if o.get("pnl", 0) > 0)
    pnl_total = sum(o.get("pnl", 0) for o in ops)
    winrate = (ganadas / total * 100) if total > 0 else 0
    pnl_str = f"{'+' if pnl_total >= 0 else ''}{pnl_total:.2f}%"
    # Calcular 'Activo desde' a partir del primer timestamp en trades, si existe
    activo_desde = "Desconocido"
    try:
        dates = []
        for o in ops:
            f = o.get("fecha") or o.get("timestamp")
            if not f:
                continue
            try:
                dates.append(datetime.fromisoformat(f))
            except Exception:
                try:
                    dates.append(datetime.strptime(f, "%Y-%m-%d %H:%M:%S"))
                except Exception:
                    pass
        if dates:
            activo_desde = min(dates).strftime("%d %b %Y %H:%M")
        else:
            # fallback: fecha de modificación de la base de datos
            try:
                mtime = os.path.getmtime(DB_NAME)
                activo_desde = datetime.fromtimestamp(mtime).strftime("%d %b %Y %H:%M")
            except Exception:
                activo_desde = "Desconocido"
    except Exception:
        activo_desde = "Desconocido"

    # Obtener balance
    try:
        print("DEBUG: Obteniendo balance para usuario:", nombre)
        balance = get_balance(nombre)
        balance_str = f"${balance:,.2f}"
    except Exception:
        balance_str = "-"

    # Obtener fecha de última modificación de la base de datos para mostrar como 'última actualización'
    try:
        mtime = os.path.getmtime(DB_NAME)
        last_upd = datetime.fromtimestamp(mtime).strftime("%d %b %Y %H:%M")
        last_upd_str = f"Última actualización: {last_upd}"
    except Exception:
        last_upd_str = ""

    return nombre, str(total), pnl_str, f"{winrate:.1f}%", activo_desde, balance_str, last_upd_str


# --- Popup confirmation for account deletion ---
from dash import ctx

@callback(
    Output("confirm-del-1", "displayed"),
    Input("btn-del-popup", "n_clicks"),
    prevent_initial_call=True,
)
def show_first_confirm(n):
    if n:
        return True
    raise PreventUpdate

@callback(
    Output("confirm-del-2", "displayed"),
    Input("confirm-del-1", "submit_n_clicks"),
    prevent_initial_call=True,
)
def show_second_confirm(n):
    if n:
        return True
    raise PreventUpdate

@callback(
    Output("auth-token", "data", allow_duplicate=True),
    Output("global-url", "pathname", allow_duplicate=True),
    Input("confirm-del-2", "submit_n_clicks"),
    prevent_initial_call=True,
)
def handle_final_delete(n):
    if n:
        return None, "/login"
    raise PreventUpdate


# Botón de cierre de sesión simple
@callback(
    Output("auth-token", "data", allow_duplicate=True),
    Output("global-url", "pathname", allow_duplicate=True),
    Input("btn-logout", "n_clicks"),
    prevent_initial_call=True,
)
def handle_logout(n):
    if not n:
        raise PreventUpdate
    return None, "/login"
