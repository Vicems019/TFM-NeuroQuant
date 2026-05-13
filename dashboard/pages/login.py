import dash
from dash import html, dcc, callback, Input, Output, State
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.auth import login_user

dash.register_page(__name__, path="/login/", name="Login")

layout = html.Div([
    html.Div([
        # Left side: Branding / Graphic
        html.Div([
            html.Div([
                html.Span("Neuro", style={"fontSize": "48px", "fontWeight": "800", "color": "white"}),
                html.Span("Quant", style={"fontSize": "48px", "fontWeight": "800", "color": "var(--blue)"}),
                html.Span(" AI", style={"fontSize": "24px", "fontWeight": "600", "color": "var(--text-muted)", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "baseline", "marginBottom": "20px"}),
            html.P("El futuro del trading algorítmico en tus manos.", style={"fontSize": "18px", "color": "var(--text-secondary)", "lineHeight": "1.5"}),
        ], style={
            "flex": "1", "padding": "60px", "background": "linear-gradient(135deg, rgba(16,24,39,1) 0%, rgba(30,58,138,0.15) 100%)",
            "display": "flex", "flexDirection": "column", "justifyContent": "center", "borderRight": "1px solid var(--border)"
        }),
        # Right side: Form
        html.Div([
            html.H2("Iniciar Sesión", style={"textAlign": "center", "marginBottom": "30px", "fontSize": "28px"}),
            html.Div([
                html.Label("Usuario", className="modal-label"),
                dcc.Input(id="login-user", type="text", className="modal-input", placeholder="Ingresa tu usuario"),
            ], className="modal-field", style={"marginBottom": "25px", "gap": "8px"}),
            html.Div([
                html.Label("Contraseña", className="modal-label"),
                html.Div([
                    dcc.Input(id="login-pass", type="password", className="modal-input", placeholder="••••••••", style={"width": "100%", "paddingRight": "35px"}),
                    html.Button("👁", id="login-show-pass", n_clicks=0, style={
                        "position": "absolute", "right": "10px", "top": "50%", "transform": "translateY(-50%)",
                        "background": "transparent", "border": "none", "color": "var(--text-secondary)", 
                        "cursor": "pointer", "fontSize": "16px"
                    })
                ], style={"position": "relative", "width": "100%"})
            ], className="modal-field", style={"marginBottom": "25px", "gap": "8px"}),
            html.Button("Acceder", id="login-btn", n_clicks=0, style={
                "width": "100%", "marginTop": "10px", "padding": "16px", "fontSize": "16px", 
                "fontWeight": "bold", "backgroundColor": "var(--blue)", "color": "white", 
                "border": "none", "borderRadius": "12px", "cursor": "pointer", 
                "boxShadow": "0 4px 15px rgba(59, 130, 246, 0.3)"
            }),
            html.Div(id="login-msg", style={"color": "var(--red)", "marginTop": "15px", "textAlign": "center", "fontSize": "14px", "fontWeight": "bold"}),
            html.Div([
                html.Span("¿No tienes cuenta? ", style={"color": "var(--text-secondary)", "fontSize": "14px"}),
                dcc.Link("Regístrate aquí", href="/register", style={"color": "var(--blue)", "fontSize": "14px", "fontWeight": "bold", "textDecoration": "none"})
            ], style={"textAlign": "center", "marginTop": "30px"})
        ], style={"flex": "1", "padding": "60px 50px", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
    ], className="panel-card", style={
        "width": "900px", "maxWidth": "95%", "margin": "10vh auto", "padding": "0", "display": "flex", "flexDirection": "row",
        "boxShadow": "0 25px 50px -12px rgba(0,0,0,0.7)", "borderRadius": "20px", "overflow": "hidden"
    })
], className="page-container", style={"display": "flex", "alignItems": "center", "minHeight": "100vh", "padding": "0"})

@callback(
    Output("login-pass", "type"),
    Output("login-show-pass", "children"),
    Input("login-show-pass", "n_clicks"),
    State("login-pass", "type"),
    prevent_initial_call=True
)
def toggle_password(n, current_type):
    if current_type == "password":
        return "text", "🙈"
    return "password", "👁"

@callback(
    Output("auth-token", "data", allow_duplicate=True),
    Output("global-url", "pathname", allow_duplicate=True),
    Output("login-msg", "children"),
    Input("login-btn", "n_clicks"),
    State("login-user", "value"),
    State("login-pass", "value"),
    prevent_initial_call=True
)
def do_login(n, user, pwd):
    if not user or not pwd:
        return dash.no_update, dash.no_update, "Completa todos los campos"
    if login_user(user, pwd):
        return user, "/home/", ""
    return dash.no_update, dash.no_update, "Usuario o contraseña incorrectos"
