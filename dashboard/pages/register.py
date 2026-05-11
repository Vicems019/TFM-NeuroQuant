import dash
from dash import html, dcc, callback, Input, Output, State
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.auth import register_user

dash.register_page(__name__, path="/register", name="Registro")

layout = html.Div([
    html.Div([
        # Left side: Branding / Graphic
        html.Div([
            html.Div([
                html.Span("Neuro", style={"fontSize": "48px", "fontWeight": "800", "color": "white"}),
                html.Span("Quant", style={"fontSize": "48px", "fontWeight": "800", "color": "var(--blue)"}),
                html.Span(" AI", style={"fontSize": "24px", "fontWeight": "600", "color": "var(--text-muted)", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "baseline", "marginBottom": "20px"}),
            html.P("Únete a la plataforma líder en inteligencia artificial para criptomonedas.", style={"fontSize": "18px", "color": "var(--text-secondary)", "lineHeight": "1.5"}),
        ], style={
            "flex": "1", "padding": "60px", "background": "linear-gradient(135deg, rgba(16,24,39,1) 0%, rgba(30,58,138,0.15) 100%)",
            "display": "flex", "flexDirection": "column", "justifyContent": "center", "borderRight": "1px solid var(--border)"
        }),
        # Right side: Form
        html.Div([
            html.H2("Registro", style={"textAlign": "center", "marginBottom": "30px", "fontSize": "28px"}),
            html.Div([
                html.Label("Usuario", className="modal-label"),
                dcc.Input(id="reg-user", type="text", className="modal-input", placeholder="Elige tu usuario"),
            ], className="modal-field", style={"marginBottom": "20px", "gap": "6px"}),
            html.Div([
                html.Label("Correo Electrónico", className="modal-label"),
                dcc.Input(id="reg-email", type="email", className="modal-input", placeholder="ejemplo@correo.com"),
            ], className="modal-field", style={"marginBottom": "20px", "gap": "6px"}),
            html.Div([
                html.Label("Nueva Contraseña", className="modal-label"),
                html.Div([
                    dcc.Input(id="reg-pass1", type="password", className="modal-input", placeholder="••••••••", style={"width": "100%", "paddingRight": "35px"}),
                    html.Button("👁", id="reg-show-pass1", n_clicks=0, style={
                        "position": "absolute", "right": "10px", "top": "50%", "transform": "translateY(-50%)",
                        "background": "transparent", "border": "none", "color": "var(--text-secondary)", 
                        "cursor": "pointer", "fontSize": "16px"
                    })
                ], style={"position": "relative", "width": "100%"})
            ], className="modal-field", style={"marginBottom": "20px", "gap": "6px"}),
            html.Div([
                html.Label("Repetir Contraseña", className="modal-label"),
                html.Div([
                    dcc.Input(id="reg-pass2", type="password", className="modal-input", placeholder="••••••••", style={"width": "100%", "paddingRight": "35px"}),
                    html.Button("👁", id="reg-show-pass2", n_clicks=0, style={
                        "position": "absolute", "right": "10px", "top": "50%", "transform": "translateY(-50%)",
                        "background": "transparent", "border": "none", "color": "var(--text-secondary)", 
                        "cursor": "pointer", "fontSize": "16px"
                    })
                ], style={"position": "relative", "width": "100%"})
            ], className="modal-field", style={"marginBottom": "20px", "gap": "6px"}),
            html.Button("Registrarse", id="reg-btn", n_clicks=0, style={
                "width": "100%", "marginTop": "10px", "padding": "16px", "fontSize": "16px", 
                "fontWeight": "bold", "backgroundColor": "var(--blue)", "color": "white", 
                "border": "none", "borderRadius": "12px", "cursor": "pointer", 
                "boxShadow": "0 4px 15px rgba(59, 130, 246, 0.3)"
            }),
            html.Div(id="reg-msg", style={"marginTop": "10px", "textAlign": "center", "fontSize": "13px"}),
            html.Div([
                html.Span("¿Ya tienes cuenta? ", style={"color": "var(--text-secondary)", "fontSize": "14px"}),
                dcc.Link("Inicia sesión aquí", href="/login", style={"color": "var(--blue)", "fontSize": "14px", "fontWeight": "bold", "textDecoration": "none"})
            ], style={"textAlign": "center", "marginTop": "20px"})
        ], style={"flex": "1", "padding": "40px 50px", "display": "flex", "flexDirection": "column", "justifyContent": "center"})
    ], className="panel-card", style={
        "width": "900px", "maxWidth": "95%", "margin": "10vh auto", "padding": "0", "display": "flex", "flexDirection": "row",
        "boxShadow": "0 25px 50px -12px rgba(0,0,0,0.7)", "borderRadius": "20px", "overflow": "hidden"
    })
], className="page-container", style={"display": "flex", "alignItems": "center", "minHeight": "100vh", "padding": "0"})

@callback(
    Output("reg-pass1", "type"),
    Output("reg-show-pass1", "children"),
    Input("reg-show-pass1", "n_clicks"),
    State("reg-pass1", "type"),
    prevent_initial_call=True
)
def toggle_p1(n, t):
    return ("text", "🙈") if t == "password" else ("password", "👁")

@callback(
    Output("reg-pass2", "type"),
    Output("reg-show-pass2", "children"),
    Input("reg-show-pass2", "n_clicks"),
    State("reg-pass2", "type"),
    prevent_initial_call=True
)
def toggle_p2(n, t):
    return ("text", "🙈") if t == "password" else ("password", "👁")

@callback(
    Output("reg-msg", "children"),
    Output("reg-msg", "style"),
    Input("reg-btn", "n_clicks"),
    State("reg-email", "value"),
    State("reg-user", "value"),
    State("reg-pass1", "value"),
    State("reg-pass2", "value"),
    prevent_initial_call=True
)
def do_register(n, email, user, p1, p2):
    if not email or not user or not p1 or not p2:
        return "Completa todos los campos", {"color": "var(--red)", "marginTop": "10px", "textAlign": "center", "fontSize": "13px"}
    if p1 != p2:
        return "Las contraseñas no coinciden", {"color": "var(--red)", "marginTop": "10px", "textAlign": "center", "fontSize": "13px"}
    
    success = register_user(user, p1, email)
    if success:
        return html.Span(["Registro exitoso. ", dcc.Link("Inicia sesión", href="/login", style={"color": "var(--blue)"})]), {"color": "var(--green)", "marginTop": "10px", "textAlign": "center", "fontSize": "13px"}
    else:
        return "El usuario ya existe", {"color": "var(--red)", "marginTop": "10px", "textAlign": "center", "fontSize": "13px"}
