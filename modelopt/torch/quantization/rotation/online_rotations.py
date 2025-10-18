"""Online activation transforms for rotation preprocessing."""

import torch
import torch.nn as nn


class OnlineHadamardTransform(nn.Module):
    """Applies Hadamard transform to activations during forward pass.

    Required for out and down projections to complete the rotation chain.
    """

    def __init__(self, had_dim: int, r2_matrix: torch.Tensor | None = None):
        """Initialize online Hadamard transform.

        Args:
            had_dim: Dimension for Hadamard (-1 for full dimension, >0 for per-chunk)
            r2_matrix: Optional pre-computed Hadamard matrix to use
        """
        super().__init__()
        self.had_dim = had_dim
        if r2_matrix is not None:
            self.register_buffer("r2_matrix", r2_matrix)
        else:
            self.r2_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard transform to input activation.

        Args:
            x: Input tensor [batch, seq_len, hidden_size] or [batch, seq_len, features]

        Returns:
            Transformed tensor with same shape
        """
        if self.had_dim == -1:
            # Full Hadamard on last dimension
            return self._apply_full_hadamard(x)
        else:
            # Per-chunk Hadamard
            return self._apply_per_chunk_hadamard(x)

    def _apply_full_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Apply full Hadamard to entire last dimension."""
        init_shape = x.shape
        hidden_size = init_shape[-1]

        # Use r2_matrix if available, otherwise generate on-the-fly
        if self.r2_matrix is not None:
            r2 = self.r2_matrix.to(x.dtype).to(x.device)
        else:
            # Generate Hadamard matrix on demand
            from .rotate_utils import get_orthogonal_matrix

            mode = "hadamard" if self._is_pow2(hidden_size) else "random"
            r2 = get_orthogonal_matrix(hidden_size, mode, x.device).to(x.dtype)

        # Reshape to 2D, apply transform, reshape back
        x_2d = x.reshape(-1, hidden_size)
        x_transformed = x_2d @ r2
        return x_transformed.reshape(init_shape)

    def _apply_per_chunk_hadamard(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard per chunk (for per-head transforms)."""
        init_shape = x.shape
        hidden_size = init_shape[-1]

        # Reshape to chunks
        x_reshaped = x.reshape(-1, hidden_size // self.had_dim, self.had_dim)

        # Apply R2 to each chunk
        if self.r2_matrix is not None:
            r2 = self.r2_matrix.to(x.dtype).to(x.device)
            x_transformed = x_reshaped @ r2
        else:
            from .rotate_utils import get_orthogonal_matrix

            r2 = get_orthogonal_matrix(self.had_dim, "hadamard", x.device).to(x.dtype)
            x_transformed = x_reshaped @ r2

        return x_transformed.reshape(init_shape)

    @staticmethod
    def _is_pow2(n: int) -> bool:
        """Check if n is power of 2."""
        return (n & (n - 1) == 0) and (n > 0)


def create_online_transform_hook(transform: OnlineHadamardTransform):
    """Create a forward pre-hook that applies Hadamard transform to input."""

    def hook(module, args):
        # args is a tuple, first element is the input tensor
        x = args[0]
        x_transformed = transform(x)
        return (x_transformed, *args[1:])

    return hook
