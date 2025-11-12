import logging
from typing import override

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from tqdm import tqdm

from simet.feature_extractor import FeatureExtractor

try: 
    from torch.amp import autocast
except ImportError: 
    try: 
        from torch.cuda.amp import autocast
    except ImportError: 
        autocast = None

logger = logging.getLogger(__name__)

class InceptionFeatureExtractor(FeatureExtractor):
    """Inception-v3 feature extractor with cached, batchwise inference.

    Loads a torchvision **Inception v3** backbone pretrained on ImageNet
    (``Inception_V3_Weights.IMAGENET1K_V1``), replaces the final fully-connected
    layer with ``torch.nn.Identity()``, and returns the penultimate features
    (dimension **2048**) for each input image. Supports CUDA with autocast
    (mixed precision) when available.

    Inputs are expected to be **already transformed** according to the
    Inception-v3 weights transforms (resize/crop to 299×299 and ImageNet
    normalization). In this project, use :class:`simet.transforms.InceptionTransform`
    to ensure compatibility.

    Args:
        cache_dir (Path | str):
            See :class:`FeatureExtractor`. Directory where feature arrays are cached.
        force_cache_recompute (bool):
            See :class:`FeatureExtractor`. If True, bypass and refresh cache.

    Attributes:
        device (torch.device):
            Device used for inference (``"cuda"`` if available, else ``"cpu"``).
        inception (torch.nn.Module):
            The initialized Inception-v3 model in eval mode, with ``fc = Identity()``.

    Notes:
        - Output feature matrix shape is ``(N, 2048)`` for a loader that yields ``N`` images.
        - Mixed precision is used automatically on CUDA if AMP is available.
        - Progress is displayed via ``tqdm`` during feature extraction.
        - Caching behavior (keying and storage) is handled by :class:`FeatureCacheService`.

    Example:
        >>> extractor = InceptionFeatureExtractor()
        >>> feats = extractor.extract_features(loader)   # may hit cache
        >>> feats.shape  # doctest: +SKIP
        (N, 2048)
    """

    @override
    def _init_model(self) -> None:
        """Initialize the pretrained Inception-v3 backbone and select device.

        Loads ImageNet-1k weights, replaces the classification head with Identity,
        switches to eval mode, and moves the model to the detected device.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device} for feature extraction")
        inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False
        )
        inception.fc = torch.nn.Identity()
        self.inception = inception.eval().to(self.device)

    @override
    @property
    def cache_key_suffix(self) -> str:
        """Suffix used to disambiguate cache entries for this extractor.

        Returns:
            str: The cache key suffix (``"inception_v3"``).
        """
        return "inception_v3"

    @override
    def _compute_features(self, loader: DataLoader[VisionDataset]) -> np.ndarray:
        """Compute a 2048-D feature vector for each image in ``loader``.

        Iterates over batches, runs the model forward pass on ``self.device``,
        and accumulates outputs. Uses AMP autocast on CUDA when available.

        Args:
            loader: DataLoader yielding ``(image, target)`` pairs. Images must be
                preprocessed for Inception-v3 (e.g., 299×299, ImageNet normalization).

        Returns:
            np.ndarray: Feature matrix of shape ``(N, 2048)`` in ``float32``.
        """
        features: list[np.ndarray] = []
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="Extracting features"):
                imgs = imgs.to(self.device, non_blocking=True)
                if autocast is not None and self.device.type == "cuda":
                    with autocast(device_type="cuda"):
                        out = self.inception(imgs)
                else:
                    out = self.inception(imgs)
                features.append(out.float().cpu().numpy())
        logger.info("Extracted features")
        return np.vstack(features)
