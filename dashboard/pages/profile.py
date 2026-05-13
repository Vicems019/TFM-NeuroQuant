import dash
from dash import html, dcc, callback, Input, Output
from dash.exceptions import PreventUpdate
from database.db_utils import get_trades, get_balance, DB_NAME
import os
from datetime import datetime

dash.register_page(__name__, path="/profile", name="Perfil")


def _row(label, value_el):
    return html.Div([
        html.Span(label, className="profile-bigcard-label"),
        html.Div(value_el, className="profile-bigcard-value"),
    ], className="profile-bigcard-row")


layout = html.Div([
    html.Div([
        # Tarjeta principal (estilo manual tipo pnl-stat-card / panel)
        html.Div([
            html.Div([
                html.Div([
                    html.Div("NQ", className="profile-bigcard-avatar"),
                ], className="profile-bigcard-avatar-wrap"),
                html.Div([
                    html.Div(id="profile-display-name", className="profile-bigcard-title"),
                    html.Div("Trader · NeuroQuant AI", className="profile-bigcard-sub"),
                    html.Div("Cuenta activa", className="profile-bigcard-badge"),
                    html.Button("Cerrar sesión", id="btn-logout", className="btn-logout", n_clicks=0),
                ], className="profile-bigcard-head-text"),
            ], className="profile-bigcard-header"),

            html.Div([
                html.Div(className="profile-asset-card", children=[
                    html.Div("Activo", className="asset-card-label"),
                    html.Div(id="profile-balance", className="asset-card-value"),
                    html.Div(id="profile-balance-upd", className="asset-card-sub"),
                ])
            ]),

            html.Div(className="profile-bigcard-divider"),
            html.Div([
                html.Div([
                    html.Div("Operaciones", className="profile-kpi-mini-label"),
                    html.Div(id="kpi-operaciones", className="profile-kpi-mini-val"),
                ], className="profile-kpi-mini"),
                html.Div([
                    html.Div("P&L total", className="profile-kpi-mini-label"),
                    html.Div(id="kpi-pnl", className="profile-kpi-mini-val"),
                ], className="profile-kpi-mini"),
                html.Div([
                    html.Div("Win rate", className="profile-kpi-mini-label"),
                    html.Div(id="kpi-winrate", className="profile-kpi-mini-val"),
                ], className="profile-kpi-mini"),
            ], className="profile-kpi-mini-row"),

            html.Div(className="profile-bigcard-divider"),

            html.Div([
                html.Div("Datos de cuenta", className="profile-bigcard-section-title"),
                _row("Correo", html.Span("usuario@tfm.es", className="profile-bigcard-val-static")),
                _row("Plan", html.Span("NeuroQuant Pro", className="profile-bigcard-val-static green")),
                _row("Activo desde", html.Span(id="profile-activo-desde", className="profile-bigcard-val-static")),
            ], className="profile-bigcard-block"),

            html.Div([
                html.Div("Modelo e inferencia", className="profile-bigcard-section-title"),
                _row("Modelo activo", html.Span("LSTM + RL PPO v2.1", className="profile-bigcard-val-static")),
                _row("Mercados", html.Span("BTC · ETH · SOL · AVAX", className="profile-bigcard-val-static")),
                _row("Actualización", html.Span("Cada 5 minutos · última sync hace ~2 min", className="profile-bigcard-val-static")),
            ], className="profile-bigcard-block"),

            html.Div(className="profile-bigcard-divider"),

            html.Div([
                html.Div("Zona de riesgo", className="profile-danger-title"),
                html.P(
                    "Eliminar la sesión en el panel. Si en el futuro hubiera cuenta persistente, "
                    "aquí se daría de baja definitivamente.",
                    className="profile-danger-copy",
                ),
                html.Div([
                    html.Div(id="wrap-del-start", style={"display": "block"}, children=[
                        html.Button(
                            "Darse de baja de la cuenta",
                            id="btn-del-start",
                            className="btn-account-delete",
                            n_clicks=0,
                        ),
                    ]),
                    html.Div(id="wrap-del-1", style={"display": "none"}, children=[
                        html.P("¿Estás seguro?", className="profile-danger-question"),
                        html.Div([
                            html.Button("Sí, continuar", id="btn-del-1-yes", className="btn-del-secondary", n_clicks=0),
                            html.Button("Cancelar", id="btn-del-1-no", className="btn-del-cancel", n_clicks=0),
                        ], className="profile-danger-actions"),
                    ]),
                    html.Div(id="wrap-del-2", style={"display": "none"}, children=[
                        html.P(
                            "¿Estás seguro de la decisión? Esta acción cerrará tu sesión en el dashboard.",
                            className="profile-danger-question",
                        ),
                        html.Div([
                            html.Button("Sí, eliminar sesión", id="btn-del-2-yes", className="btn-account-delete", n_clicks=0),
                            html.Button("Cancelar", id="btn-del-2-no", className="btn-del-cancel", n_clicks=0),
                        ], className="profile-danger-actions"),
                    ]),
                ], className="profile-danger-stack"),
            ], className="profile-danger-zone"),

        ], className="profile-bigcard"),

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
        balance = get_balance()
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


@callback(
    Output("wrap-del-start", "style"),
    Output("wrap-del-1", "style"),
    Output("wrap-del-2", "style"),
    Output("auth-token", "data", allow_duplicate=True),
    Output("global-url", "pathname", allow_duplicate=True),
    Input("btn-del-start", "n_clicks"),
    Input("btn-del-1-yes", "n_clicks"),
    Input("btn-del-1-no", "n_clicks"),
    Input("btn-del-2-yes", "n_clicks"),
    Input("btn-del-2-no", "n_clicks"),
    prevent_initial_call=True,
)
def account_delete_flow(ns, n1y, n1n, n2y, n2n):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    tid = ctx.triggered_id
    blk = {"display": "block"}
    non = {"display": "none"}

    if tid == "btn-del-start":
        return non, blk, non, dash.no_update, dash.no_update
    if tid == "btn-del-1-no":
        return blk, non, non, dash.no_update, dash.no_update
    if tid == "btn-del-1-yes":
        return non, non, blk, dash.no_update, dash.no_update
    if tid == "btn-del-2-no":
        return blk, non, non, dash.no_update, dash.no_update
    if tid == "btn-del-2-yes":
        return blk, non, non, None, "/login"
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
