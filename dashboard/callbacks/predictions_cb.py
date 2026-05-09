from dash import Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash

def register_callbacks(app, pipeline: PredictionPipeline):

    # ─── Store central de estado ──────────────────────────────────────────────
    # En el layout: dcc.Store(id='app-state', storage_type='session')
    # Evita re-renders innecesarios; solo cambia lo que cambia

    @app.callback(
        Output('app-state', 'data'),
        Input('currency-selector', 'value'),
        State('app-state', 'data'),
        prevent_initial_call=True
    )
    def on_currency_change(new_symbol, current_state):
        """Cambio de divisa: solo actualiza el estado, no recalcula todo."""
        if not new_symbol:
            raise PreventUpdate
        
        current_state = current_state or {}
        
        # Invalida caché del símbolo anterior
        old_symbol = current_state.get('symbol')
        if old_symbol and old_symbol != new_symbol:
            invalidate_symbol(old_symbol)
        
        return {**current_state, 'symbol': new_symbol, 'needs_refresh': True}

    @app.callback(
        [Output('prediction-graph', 'figure'),
         Output('decision-badge', 'children'),
         Output('metrics-store', 'data'),
         Output('loading-state', 'data')],
        Input('app-state', 'data'),
        State('market-data-store', 'data'),
        background=True,                    # ← Dash 2.6+: callback en background
        running=[
            (Output('loading-overlay', 'style'), 
             {'display': 'block'}, {'display': 'none'}),
        ],
        prevent_initial_call=True
    )
    def update_predictions(state, market_data):
        """
        Callback en background: no bloquea la UI durante predicción.
        Requiere: pip install dash[diskcache] celery
        """
        if not state or not state.get('needs_refresh'):
            raise PreventUpdate
        
        symbol = state['symbol']
        
        try:
            data_array = np.array(market_data['prices'])
            result = pipeline.run(symbol, data_array)
            
            figure = build_prediction_figure(result)
            decision_badge = build_decision_badge(result.decision, result.action_probs)
            metrics_data = result.metrics
            
            return figure, decision_badge, metrics_data, {'status': 'ok', 
                                                           'latency': result.latency_ms}
        
        except Exception as e:
            logger.error(f"Error en predicción {symbol}: {e}")
            return no_update, no_update, no_update, {'status': 'error', 'msg': str(e)}

    @app.callback(
        Output('metrics-display', 'children'),
        Input('metrics-store', 'data'),
    )
    def update_metrics_display(metrics):
        """Métricas se actualizan independientemente de la predicción."""
        if not metrics:
            raise PreventUpdate
        return build_metrics_cards(metrics)