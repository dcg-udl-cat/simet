import torch
import torch.nn as nn


class RocMLP(nn.Module):
    """Two-hidden-layer MLP that outputs a single **logit** for binary tasks.

    Architecture:
        in_dim → Linear(h1) → ReLU → Dropout(p)
               → Linear(h2) → ReLU → Dropout(p)
               → Linear(1)  → (logit)

    Intended for binary classification / scoring (e.g., ROC-AUC evaluation) using
    **logits** (apply `torch.nn.BCEWithLogitsLoss` or `torch.sigmoid` externally).

    Args:
        in_dim (int): Input feature dimension `D`.
        h1 (int, optional): Hidden size of the first layer. Defaults to 256.
        h2 (int, optional): Hidden size of the second layer. Defaults to 128.
        p (float, optional): Dropout probability applied after each hidden ReLU.
            Defaults to 0.2.

    Attributes:
        net (nn.Sequential): The MLP stack producing a single logit.

    Input/Output:
        - Input:  `x` of shape `(N, D)` with `D == in_dim`.
        - Output: Logits of shape `(N,)` (after `squeeze(-1)`).

    Example:
        >>> model = RocMLP(in_dim=2048).eval()
        >>> x = torch.randn(8, 2048)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8])
    """

    def __init__(self, in_dim: int, h1: int = 256, h2: int = 128, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(h2, 1),  # logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of feature vectors.

        Args:
            x (torch.Tensor): Input tensor of shape `(N, in_dim)`.

        Returns:
            torch.Tensor: Logits of shape `(N,)`. Apply `sigmoid` for probabilities
            or use with `BCEWithLogitsLoss` for numerically stable training.
        """
        return self.net(x).squeeze(-1)  # logits
