import torch
import threading
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton thread-safe para gestión de modelos."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, torch.nn.Module] = {}
        self._model_locks: Dict[str, threading.Lock] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True
        logger.info(f"ModelManager inicializado en device: {self._device}")

    def preload_models(self, model_configs: Dict[str, dict]):
        """
        Precarga todos los modelos al arrancar la app.
        
        Args:
            model_configs: {
                'BTC_LSTM': {'path': '...', 'class': LSTMModel, 'params': {...}},
                'BTC_SAC':  {'path': '...', 'class': SACAgent, 'params': {...}},
            }
        """
        threads = []
        for model_id, config in model_configs.items():
            t = threading.Thread(
                target=self._load_single_model,
                args=(model_id, config),
                daemon=True
            )
            threads.append(t)
            t.start()
        
        # Esperar a que todos carguen
        for t in threads:
            t.join()
        
        logger.info(f"✅ Modelos precargados: {list(self._models.keys())}")

    def _load_single_model(self, model_id: str, config: dict):
        """Carga un modelo de forma segura."""
        try:
            model_class = config['class']
            model = model_class(**config.get('params', {}))
            
            state_dict = torch.load(
                config['path'],
                map_location=self._device,
                weights_only=True  # Seguridad: evita pickle arbitrario
            )
            model.load_state_dict(state_dict)
            model.eval()
            model.to(self._device)
            
            # Warm-up: primera inferencia es siempre más lenta
            self._warmup_model(model, config)
            
            self._models[model_id] = model
            self._model_locks[model_id] = threading.Lock()
            logger.info(f"✅ {model_id} cargado y warm-up completado")
            
        except Exception as e:
            logger.error(f"❌ Error cargando {model_id}: {e}")

    def _warmup_model(self, model, config):
        """Ejecuta inferencia dummy para compilar kernels CUDA."""
        with torch.no_grad():
            dummy = torch.zeros(1, config.get('seq_len', 60), 
                              config.get('features', 5)).to(self._device)
            model(dummy)

    def get_model(self, model_id: str) -> Optional[torch.nn.Module]:
        """Obtiene modelo thread-safe."""
        return self._models.get(model_id)

    def is_ready(self, model_id: str) -> bool:
        return model_id in self._models

    @property
    def all_loaded(self) -> bool:
        return len(self._models) > 0


# Instancia global
model_manager = ModelManager()