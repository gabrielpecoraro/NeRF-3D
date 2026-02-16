import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from the NeRF paper.

    Maps a D-dimensional input to a (D + D * 2 * num_frequencies)-dimensional
    output using sin/cos at exponentially increasing frequencies.

    For xyz (L=10): 3 -> 63 dimensions.
    For direction (L=4): 3 -> 27 dimensions.
    """

    def __init__(self, num_frequencies: int, include_identity: bool = True):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_identity = include_identity

        freq_bands = 2.0 ** torch.arange(num_frequencies)  # [1, 2, 4, ..., 2^(L-1)]
        self.register_buffer("freq_bands", freq_bands)

    @property
    def output_dim(self) -> int:
        """Output dimensionality for a 3-D input."""
        d = 3
        out = d * 2 * self.num_frequencies
        if self.include_identity:
            out += d
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., D) input coordinates.

        Returns:
            (..., output_dim) positional-encoded features.
        """
        encodings = []
        if self.include_identity:
            encodings.append(x)

        for freq in self.freq_bands:
            encodings.append(torch.sin(freq * x))
            encodings.append(torch.cos(freq * x))

        return torch.cat(encodings, dim=-1)
