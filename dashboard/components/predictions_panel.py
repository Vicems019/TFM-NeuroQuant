from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

def render():
    return html.Div([
        html.H3("Predicciones LSTM — próximas 4h"),
        html.Div(id="tabla-predicciones"),   # Se rellena por callback
        html.Div(id="panel-decision-rl"),    # COMPRAR / VENDER / HOLD
        dbc.Button("Actualizar predicción ↗", id="btn-actualizar", 
                   className="btn-actualizar", n_clicks=0),
    ], className="predictions-panel")


@callback(
    Output("tabla-predicciones", "children"),
    Input("store-predicciones", "data"),
    Input("store-cripto", "data"),
)
def render_tabla(predicciones, cripto):
    if not predicciones:
        return html.P("Cargando...")
    
    precio_actual = predicciones.get("precio_actual", 0)
    filas = []
    for h, label in [(1,"1h"),(2,"2h"),(3,"3h"),(4,"4h")]:
        precio = predicciones[label]
        cambio = (precio - precio_actual) / precio_actual * 100
        color = "#e74c3c" if cambio < 0 else "#2ecc71"
        filas.append(html.Div([
            html.Span(f"en {label}"),
            html.Span(f"${precio:,.3f}"),
            html.Span(f"{cambio:+.2f}%", style={"color": color, "background": color+"22",
                                                  "padding": "2px 8px", "borderRadius": "4px"})
        ], className="fila-prediccion"))
    return filas


@callback(
    Output("panel-decision-rl", "children"),
    Input("store-decision", "data"),
)
def render_decision(decision):
    if not decision:
        return html.Div()
    
    accion = decision.get("accion", "HOLD")
    colores = {"COMPRAR": "#27ae60", "VENDER": "#922b21", "HOLD": "#7f8c8d"}
    iconos  = {"COMPRAR": "🟢", "VENDER": "🔴", "HOLD": "⚪"}
    
    return html.Div([
        html.Small("Decisión RL (PPO)"),
        html.Div([
            html.Span(iconos[accion]),
            html.Span(accion, style={"fontWeight": "bold", "fontSize": "1.3rem"})
        ])
    ], style={"background": colores[accion], "borderRadius": "8px", "padding": "16px"})