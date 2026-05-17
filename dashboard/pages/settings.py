import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/settings", name="Ajustes")

layout = html.Div([
    html.Div([
        html.Div("⚙ Ajustes del sistema", className="page-hero-title"),

        # Sección modelo
        html.Div([
            html.Div("Modelo de predicción", className="settings-section-title"),
            html.Div([
                html.Div([
                    html.Label("Horizonte de predicción", className="sett-label"),
                    dcc.Dropdown(
                        options=[{"label": f"{h}h", "value": h} for h in [1, 2, 4, 8, 12, 24]],
                        value=4, clearable=False, className="modal-dropdown", id="sett-horizonte"
                    ),
                ], className="sett-field"),
                html.Div([
                    html.Label("Intervalo de actualización", className="sett-label"),
                    dcc.Dropdown(
                        options=[{"label": f"{m} min", "value": m} for m in [1, 5, 15, 30, 60]],
                        value=5, clearable=False, className="modal-dropdown", id="sett-intervalo"
                    ),
                ], className="sett-field"),
                html.Div([
                    html.Label("Umbral de confianza mínimo (%)", className="sett-label"),
                    dcc.Slider(id="sett-confianza", min=50, max=95, step=5, value=70,
                               marks={i: f"{i}%" for i in range(50, 100, 10)},
                               className="sett-slider"),
                ], className="sett-field"),
            ], className="sett-fields-grid"),
        ], className="settings-section panel-card"),

        # Sección Visualización (Moneda)
        html.Div([
            html.Div("Visualización", className="settings-section-title"),
            html.Div([
                html.Div([
                    html.Label("Divisa principal", className="sett-label"),
                    dcc.Dropdown(
                        id="sett-currency",
                        options=[
                            {"label": "🇺🇸 Dólar Estadounidense (USD)", "value": "USD"},
                            {"label": "🇪🇺 Euro (EUR)", "value": "EUR"},
                            {"label": "🇬🇧 Libra Esterlina (GBP)", "value": "GBP"},
                            {"label": "🇯🇵 Yen Japonés (JPY)", "value": "JPY"},
                        ],
                        value="USD", clearable=False, className="modal-dropdown",
                        persistence=True, persistence_type="local"
                    ),
                ], className="sett-field"),
            ], className="sett-fields-grid"),
        ], className="settings-section panel-card"),

        # Sección notificaciones
        html.Div([
            html.Div("Notificaciones", className="settings-section-title"),
            html.Div([
                html.Div([
                    html.Span("Alertas de señal de compra/venta", className="toggle-label"),
                    dcc.Checklist(options=[{"label": "", "value": "on"}], value=["on"],
                                  id="sett-alerta-señal", className="toggle-check"),
                ], className="toggle-row"),
                html.Div([
                    html.Span("Alerta de drawdown máximo", className="toggle-label"),
                    dcc.Checklist(options=[{"label": "", "value": "on"}], value=[],
                                  id="sett-alerta-dd", className="toggle-check"),
                ], className="toggle-row"),
                html.Div([
                    html.Span("Resumen diario por email", className="toggle-label"),
                    dcc.Checklist(options=[{"label": "", "value": "on"}], value=["on"],
                                  id="sett-alerta-email", className="toggle-check"),
                ], className="toggle-row"),
            ]),
        ], className="settings-section panel-card"),

        # Guardar
        html.Div([
            html.Button("Guardar cambios", id="btn-save-settings", className="btn-save", n_clicks=0),
            html.Div(id="save-feedback", className="save-feedback"),
        ], className="sett-actions"),

    ], className="page-container"),
], className="page-container")


@callback(
    Output("save-feedback", "children"),
    Input("btn-save-settings", "n_clicks"),
    prevent_initial_call=True,
)
def save_settings(n):
    return "✓ Configuración guardada correctamente" if n else ""
