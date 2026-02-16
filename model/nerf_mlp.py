import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NeRFMLP(nn.Module):
    """Classic NeRF MLP with hierarchical density/color prediction.

    Architecture:
        - 8 fully-connected layers (256 units, ReLU)
        - Skip connection at layer 4 (concat positional-encoded xyz)
        - Density head: Linear(256, 1) — raw sigma
        - Color head: feature(256) + dir_enc -> Linear(283, 128) + ReLU -> Linear(128, 3) + Sigmoid

    Layer dimensions:
        fc_0: (pos_enc_dim, 256)
        fc_1-3: (256, 256)
        fc_4: (256 + pos_enc_dim, 256)  [skip connection]
        fc_5-7: (256, 256)
        sigma_head: (256, 1)
        feature_head: (256, 256)
        color_hidden: (256 + dir_enc_dim, 128)
        color_out: (128, 3)
    """

    def __init__(
        self,
        pos_enc_dim: int = 63,
        dir_enc_dim: int = 27,
        hidden_dim: int = 256,
        num_layers: int = 8,
        skip_layer: int = 4,
        color_hidden_dim: int = 128,
    ):
        super().__init__()
        self.skip_layer = skip_layer

        # Build the main MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                in_dim = pos_enc_dim
            elif i == skip_layer:
                in_dim = hidden_dim + pos_enc_dim
            else:
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers = nn.ModuleList(layers)

        # Density output (raw sigma, no activation — ReLU applied during rendering)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Feature bottleneck for color prediction
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)

        # Color branch: direction-dependent
        self.color_hidden = nn.Linear(hidden_dim + dir_enc_dim, color_hidden_dim)
        self.color_out = nn.Linear(color_hidden_dim, 3)

    def forward(
        self, pos_encoded: torch.Tensor, dir_encoded: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the NeRF MLP.

        Args:
            pos_encoded: (batch, pos_enc_dim) positional-encoded 3D coordinates.
            dir_encoded: (batch, dir_enc_dim) positional-encoded view directions.

        Returns:
            rgb: (batch, 3) predicted color in [0, 1].
            sigma: (batch, 1) predicted raw density.
        """
        h = pos_encoded
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = F.relu(layer(h))

        sigma = self.sigma_head(h)

        feature = self.feature_head(h)
        color_input = torch.cat([feature, dir_encoded], dim=-1)
        color = F.relu(self.color_hidden(color_input))
        rgb = torch.sigmoid(self.color_out(color))

        return rgb, sigma
