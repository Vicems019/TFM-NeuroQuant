import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import SAC

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=4, mlp_hidden=64, head_hidden=32, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # input_proj.0 -> Linear, input_proj.1 -> LayerNorm
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.attn = nn.Linear(hidden_size, 1)
        self.norm = nn.LayerNorm(hidden_size)
        
        # mlp.0, mlp.1, mlp.4 (Linear, LayerNorm, Linear)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, head_hidden)
        )
        
        self.head = nn.Linear(head_hidden, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        
        # Attention
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(weights * lstm_out, dim=1)
        context = self.norm(context)
        
        x = self.mlp(context)
        return self.head(x)

class SACAgent:
    """Wrapper para el agente Soft Actor-Critic de Stable Baselines 3."""
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = None
        self.device = "cpu"

    def load(self, path, device="cpu"):
        self.device = device
        self.model = SAC.load(path, device=device)
        return self

    def eval(self):
        # Para compatibilidad con torch.nn.Module.eval()
        pass

    def to(self, device):
        self.device = device
        if self.model:
            self.model.policy.to(device)
        return self

    def load_state_dict(self, state_dict):
        # SB3 modelos no usan load_state_dict de la misma forma
        pass

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if self.model is None:
            return np.zeros(self.action_dim)
        
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        action, _ = self.model.predict(x, deterministic=True)
        return action

    def get_action_probs(self, x):
        """
        Simula probabilidades de acción a partir de la salida continua del SAC.
        Mapea el rango [-1, 1] a [SELL, HOLD, BUY].
        """
        if self.model is None:
            return torch.tensor([0.33, 0.33, 0.34]).to(self.device)
            
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            
            # El actor devuelve la media de la acción en SAC
            action_mean = self.model.policy.actor(x)
            action_val = action_mean.item() if action_mean.numel() == 1 else action_mean[0].item()
            
            # Mapeo basado en los umbrales del notebook (-0.33, 0.33)
            probs = np.zeros(3)
            if action_val < -0.33:
                probs[0] = 0.8  # SELL
                probs[1] = 0.15
                probs[2] = 0.05
            elif action_val > 0.33:
                probs[2] = 0.8  # BUY
                probs[1] = 0.15
                probs[0] = 0.05
            else:
                probs[1] = 0.7  # HOLD
                probs[0] = 0.15
                probs[2] = 0.15
                
            return torch.FloatTensor(probs).to(self.device)
