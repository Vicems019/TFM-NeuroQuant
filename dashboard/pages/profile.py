import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
dash.register_page(__name__, path="/profile", name="Perfil")
layout = html.Div([
    html.Div([
        # Avatar
        html.Div([
            html.Div("👤", className="profile-avatar"),
            html.Div([
                html.Div("Usuario TFM", className="profile-name"),
                html.Div("Trader · NeuroQuant AI", className="profile-role"),
            ]),
        ], className="profile-hero"),
        # Stats row
        html.Div([
            html.Div([html.Div("42", className="pstat-val green"),   html.Div("Operaciones", className="pstat-label")], className="pstat-card"),
            html.Div([html.Div("+18.4%", className="pstat-val green"), html.Div("P&L Total",    className="pstat-label")], className="pstat-card"),
            html.Div([html.Div("68%",    className="pstat-val amber"), html.Div("Win Rate",     className="pstat-label")], className="pstat-card"),
            html.Div([html.Div("1.87",   className="pstat-val"),       html.Div("Sharpe",       className="pstat-label")], className="pstat-card"),
        ], className="profile-stats"),
        # Info card
        html.Div([
            html.Div("Información de la cuenta", className="section-title", style={"marginBottom": "16px"}),
            html.Div([
                html.Div([html.Span("Email",         className="info-label"), html.Span("usuario@tfm.es",     className="info-val")], className="info-row"),
                html.Div([html.Span("Plan",          className="info-label"), html.Span("NeuroQuant Pro",      className="info-val green")], className="info-row"),
                html.Div([html.Span("Activo desde",  className="info-label"), html.Span("Enero 2024",          className="info-val")], className="info-row"),
                html.Div([html.Span("Modelo activo", className="info-label"), html.Span("LSTM + RL PPO v2.1",  className="info-val")], className="info-row"),
            ], className="info-table"),
        ], className="panel-card"),
    ], className="page-container profile-page"),
], className="page-container")
