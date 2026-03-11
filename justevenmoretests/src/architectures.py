import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TabularBase(nn.Module):
    """Basisklasse mit gemeinsamer predict_proba-Logik.

    FIX #12: Batched Inference — kein vollständiger Datensatz auf GPU.
    Eliminiert Code-Duplizierung zwischen MLP, ResNet und FTT.
    """

    def predict_proba(self, x, bs=1024):
        self.eval()
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        results = []
        with torch.no_grad():
            for i in range(0, len(x), bs):
                xb = x[i:i + bs].to(device)
                results.append(torch.softmax(self.forward(xb), 1).cpu().numpy())

        return np.concatenate(results, axis=0)

    def get_features_batched(self, x, bs=1024):
        """Batched Feature-Extraktion (CPU-Numpy oder GPU-Tensor als Input)."""
        self.eval()
        device = next(self.parameters()).device

        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)

        results = []
        with torch.no_grad():
            for i in range(0, len(x), bs):
                xb = x[i:i + bs].to(device)
                results.append(self.get_features(xb).cpu())

        return torch.cat(results, dim=0)


# --- MLP ---
class MLP(TabularBase):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, n_classes=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.feat = nn.Sequential(*layers)
        self.clf = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        return self.clf(self.feat(x))

    def get_features(self, x):
        return self.feat(x)


# --- ResNet ---
class ResBlock(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(dropout),
            nn.BatchNorm1d(d), nn.ReLU(), nn.Linear(d, d), nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetTabular(TabularBase):
    def __init__(self, input_dim, output_dim, n_layers=2, d_model=128, dropout=0.3):
        super().__init__()
        self.emb = nn.Linear(input_dim, d_model)
        self.blocks = nn.Sequential(*[ResBlock(d_model, dropout) for _ in range(n_layers)])
        self.norm = nn.BatchNorm1d(d_model)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x):
        return self.head(self.get_features(x))

    def get_features(self, x):
        return F.relu(self.norm(self.blocks(self.emb(x))))


# --- FT-Transformer ---
class FTTransformer(TabularBase):
    def __init__(self, n_features, d_token=192, n_layers=3, n_heads=8,
                 d_ffn_factor=1.33, dropout=0.1, n_classes=2):
        super().__init__()
        self.tokenizer_w = nn.Parameter(torch.randn(n_features, d_token))
        self.tokenizer_b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.tokenizer_w)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads,
            dim_feedforward=int(d_token * d_ffn_factor),
            dropout=dropout, activation="relu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_norm = nn.LayerNorm(d_token)
        self.head = nn.Linear(d_token, n_classes)

    def get_features(self, x):
        x_emb = x.unsqueeze(-1) * self.tokenizer_w + self.tokenizer_b
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x_seq = torch.cat((cls, x_emb), dim=1)
        x_out = self.transformer(x_seq)
        return self.head_norm(x_out[:, 0, :])

    def forward(self, x):
        return self.head(self.get_features(x))
