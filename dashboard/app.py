import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,               # Activa el sistema de pages
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
    title="Crypto LSTM + RL Dashboard"
)

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="store-cripto", data="BTC"),      # Estado global: cripto activa
    dcc.Store(id="store-predicciones", data={}),   # Cache de predicciones LSTM
    dcc.Store(id="store-decision", data={}),        # Cache de decisión RL
    dcc.Interval(id="intervalo-auto", interval=5*60*1000, n_intervals=0),  # Refresco 5min
    dash.page_container                            # Renderiza la página activa
])

if __name__ == "__main__":
    app.run(debug=True)