import dash
from dash import html, dcc, callback, Input, Output
from database.db_utils import get_trades

dash.register_page(__name__, path="/profile", name="Perfil")


def _info_row(label, val=None, icon="", row_id=None, val_cls=""):
    val_el = html.Span(id=row_id, className=f"info-val {val_cls}".strip()) if row_id else \
             html.Span(val or "—", className=f"info-val {val_cls}".strip())
    return html.Div([
        html.Span(icon, className="info-row-icon"),
        html.Span(label, className="info-label"),
        val_el,
    ], className="info-row")


layout = html.Div([
    html.Div([

        # ── Hero: avatar + nombre + rol ──────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("👤", className="profile-avatar-icon"),
            ], className="profile-avatar-wrap"),
            html.Div([
                html.Div(id="profile-username", className="profile-hero-name"),
                html.Div("Trader · NeuroQuant AI", className="profile-hero-role"),
                html.Div([
                    html.Span("● ", style={"color": "#10b981", "fontSize": "10px"}),
                    html.Span("Activo", className="profile-status-text"),
                ], className="profile-status-badge"),
            ], className="profile-hero-info"),
        ], className="profile-hero-card"),

        # ── KPIs ─────────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div("📊", className="kpi-icon"),
                html.Div(id="kpi-operaciones", className="kpi-val"),
                html.Div("Operaciones", className="kpi-label"),
            ], className="kpi-card"),
            html.Div([
                html.Div("📈", className="kpi-icon"),
                html.Div(id="kpi-pnl", className="kpi-val green"),
                html.Div("P&L Total", className="kpi-label"),
            ], className="kpi-card"),
            html.Div([
                html.Div("🎯", className="kpi-icon"),
                html.Div(id="kpi-winrate", className="kpi-val amber"),
                html.Div("Win Rate", className="kpi-label"),
            ], className="kpi-card"),
            html.Div([
                html.Div("⚡", className="kpi-icon"),
                html.Div("1.87", className="kpi-val"),
                html.Div("Sharpe Ratio", className="kpi-label"),
            ], className="kpi-card"),
        ], className="profile-kpis-grid"),

        # ── Dos columnas: Cuenta + Modelo ────────────────────────────────────
        html.Div([

            # Columna izquierda: Información de la cuenta
            html.Div([
                html.Div([
                    html.Span("🔐", className="panel-icon"),
                    html.Span("Cuenta", className="panel-title"),
                ], className="panel-header"),
                html.Div([
                    _info_row("Usuario",      row_id="profile-info-user", icon="👤"),
                    _info_row("Email",        val="usuario@tfm.es",       icon="✉️"),
                    _info_row("Plan",         val="NeuroQuant Pro",        icon="⭐", val_cls="green"),
                    _info_row("Activo desde", val="Enero 2024",           icon="📅"),
                ], className="info-panel-body"),
            ], className="profile-panel"),

            # Columna derecha: Configuración del modelo
            html.Div([
                html.Div([
                    html.Span("🤖", className="panel-icon"),
                    html.Span("Modelo IA", className="panel-title"),
                ], className="panel-header"),
                html.Div([
                    _info_row("Modelo activo", val="LSTM + RL PPO v2.1",   icon="🧠"),
                    _info_row("Criptos",       val="BTC · ETH · SOL · AVAX", icon="💎"),
                    _info_row("Frecuencia",    val="Cada 5 minutos",        icon="🔄"),
                    _info_row("Última sync",   val="Hace 2 min",            icon="🕐"),
                ], className="info-panel-body"),
            ], className="profile-panel"),

        ], className="profile-panels-row"),

    ], className="profile-content"),
], className="page-container")


# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@callback(
    Output("profile-username",  "children"),
    Output("profile-info-user", "children"),
    Output("kpi-operaciones",   "children"),
    Output("kpi-pnl",           "children"),
    Output("kpi-winrate",       "children"),
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
    return nombre, nombre, str(total), pnl_str, f"{winrate:.1f}%"
