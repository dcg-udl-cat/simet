from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from simet.services import FeatureCacheService


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction with transparent caching.

    This class defines the interface and common behavior for computing feature
    matrices from a `torch.utils.data.DataLoader`. It wraps a cache service so
    callers can reuse previously-computed features without changing their code.

    Args:
        cache_dir (Path | str):
            Directory used by the cache to store computed features. Defaults to
            ``"cache/features"`` relative to the working directory.
        force_cache_recompute (bool):
            If ``True``, bypasses any existing cached artifact and recomputes
            features, then updates the cache. Defaults to ``False``.

    Attributes:
        cache_service (FeatureCacheService):
            Helper used to read/write cached feature arrays.
        force_cache_recompute (bool):
            Whether to recompute features even if a cache entry exists.

    Subclassing:
        Subclasses must implement `_init_model`, `_compute_features`, and
        the `cache_key_suffix` property.

        - `_init_model()`:
            Initialize any model state (e.g., load weights, select device, set
            eval mode). Called once from `__init__`.
        - `_compute_features(loader)`:
            Compute and return a NumPy array of features for *all* samples yielded
            by `loader`. The method should not manage caching.
            Return shape is typically ``(N, D)`` where:
              * ``N`` is the number of samples in the loader,
              * ``D`` is the feature dimension produced by the extractor.
        - `cache_key_suffix`:
            A short string identifying the extractor configuration (e.g.,
            ``"inception_v3_pool3"`` or a hash of model+preprocessing). Used
            to disambiguate cache entries across different extractors.

    Example:
        >>> extractor = MyExtractor(cache_dir="cache/features")
        >>> feats = extractor.extract_features(loader)   # may hit cache
        >>> feats.shape  # doctest: +SKIP
        (N, D)
    """

    def __init__(
        self,
        cache_dir: Path = Path("cache/features"),
        force_cache_recompute: bool = False,
    ) -> None:
        """Initialize the cache service and the underlying model.

        Calls `_init_model()` so subclasses can set up weights/devices.

        Args:
            cache_dir: Directory where feature arrays will be cached.
            force_cache_recompute: If True, forces recomputation even when a
                cache entry for the given (loader, suffix) exists.
        """
        self.cache_service = FeatureCacheService(cache_dir)
        self.force_cache_recompute = force_cache_recompute
        self._init_model()

    @abstractmethod
    def _init_model(self) -> None:
        """Initialize model resources (e.g., load weights, set device/mode).

        Implementations should be side-effect free beyond model setup.
        Do not perform feature computation here.
        """
        pass

    @abstractmethod
    def _compute_features(self, loader: DataLoader[VisionDataset]) -> np.ndarray:
        """Compute feature matrix for all samples in `loader`.

        Implementations should iterate over `loader`, run forward passes, and
        return a NumPy array of shape ``(N, D)``. The method should be pure with
        respect to caching (i.e., compute only; no reads/writes to cache).

        Args:
            loader: A DataLoader yielding samples compatible with this extractor.

        Returns:
            np.ndarray: Feature matrix of shape ``(N, D)`` (float dtype recommended).
        """
        pass

    @property
    @abstractmethod
    def cache_key_suffix(self) -> str:
        """Short string used to differentiate cache entries.

        Should uniquely identify the extractor configuration that affects the
        resulting features (e.g., backbone name, layer, preprocessing variant).
        """
        pass

    def extract_features(self, loader: DataLoader[VisionDataset]) -> np.ndarray:
        """Return features for `loader`, using cache when available.

        Delegates to `FeatureCacheService.get_or_compute`, which constructs a
        cache key based on the loader identity and `cache_key_suffix`. When
        `force_cache_recompute` is True, the cache is bypassed and updated.

        Args:
            loader: DataLoader providing the dataset whose features are required.

        Returns:
            np.ndarray: Feature matrix of shape ``(N, D)``.
        """
        return self.cache_service.get_or_compute(
            loader=loader,
            compute_fn=self._compute_features,
            cache_key_suffix=self.cache_key_suffix,
            force_recompute=self.force_cache_recompute,
        )
