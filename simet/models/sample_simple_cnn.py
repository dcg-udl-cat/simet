import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class SimpleCNN(nn.Module):
    """A small convolutional classifier (C64 head with global average pooling).

    Architecture:
        features:
            Conv2d(3→32, k=3, p=1) → ReLU → MaxPool2d(2)
            Conv2d(32→64, k=3, p=1) → ReLU → MaxPool2d(2)
        global_pool:
            AdaptiveAvgPool2d(output_size=1)
        classifier:
            Flatten → Linear(64→128) → ReLU → Dropout(p=0.5) → Linear(128→num_classes)

    Input/Output:
        - Input tensor shape: ``(N, 3, H, W)`` with H and W ≥ 4 (multiples of 4 recommended
          due to two 2×2 pools). Typical usage is 64×64 images normalized to a reasonable range.
        - Output tensor shape: ``(N, num_classes)`` (unnormalized logits).

    Args:
        num_classes (int): Number of output classes (size of the final logits vector).
            Defaults to 10.

    Attributes:
        features (nn.Sequential): Two conv blocks with ReLU and 2×2 max pooling.
        global_pool (nn.AdaptiveAvgPool2d): Global average pooling to 1×1 per channel.
        classifier (nn.Sequential): MLP head producing class logits.

    Example:
        >>> model = SimpleCNN(num_classes=2).eval()
        >>> x = torch.randn(8, 3, 64, 64)
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 2])
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: Logits of shape ``(N, num_classes)``. Apply `softmax` or
            `log_softmax` externally if probabilities are needed.
        """
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)
