import hashlib
import logging
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__)


class FeatureCacheService:
    """Disk cache for feature matrices computed from DataLoaders.

    Stores/loads precomputed feature arrays (e.g., `(N, D)` floats) keyed by a
    deterministic hash derived from:
      - dataset identity (root path or dataset type/length),
      - subset membership (sorted indices for `torch.utils.data.Subset`),
      - feature-extractor suffix (e.g., `"inception_v3"`),
      - loader parameters (e.g., `batch_size`).

    Features are serialized via `pickle` under `cache_dir/<md5>.pkl`.

    Args:
        cache_dir (Path | str):
            Directory where cached feature files are written/read.
            Created if it does not exist. Defaults to `"cache/features"`.

    Example:
        >>> svc = FeatureCacheService("cache/features")
        >>> feats = svc.get_or_compute(
        ...     loader=my_loader,
        ...     compute_fn=my_extractor_fn,   # def f(loader) -> np.ndarray
        ...     cache_key_suffix="inception_v3",
        ... )
        >>> feats.shape  # doctest: +SKIP
        (N, D)

    Notes:
        - **Security**: `pickle` is not safe for untrusted inputs. Only load files
          created by this application.
        - **Subsets**: For `Subset` datasets, the cache key includes a hash of the
          **sorted indices**, making different subsets cache to different files.
        - **Invalidation**: Changing any key component (dataset path/size, suffix,
          batch size) produces a different cache key and thus a cache miss.
    """

    def __init__(self, cache_dir: Path = Path("cache/features")) -> None:
        """Initialize the cache directory and logger.

        Args:
            cache_dir: Directory to store cache files (`*.pkl`).
        """
        logger.debug(f"Initializing FeatureCacheService with cache_dir: {cache_dir}")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_or_compute(
        self,
        loader: DataLoader[VisionDataset],
        compute_fn: Callable[[DataLoader[VisionDataset]], np.ndarray],
        cache_key_suffix: str = "",
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Return cached features for `loader` or compute and cache them.

        Builds a stable cache key (see `_generate_cache_key`) and attempts to
        load the feature array. If missing or `force_recompute=True`, runs
        `compute_fn(loader)`, saves the result, and returns it.

        Args:
            loader: DataLoader providing samples for feature extraction.
            compute_fn: Callable that computes the feature matrix from `loader`
                and returns a NumPy array, typically shape `(N, D)`.
            cache_key_suffix: Disambiguator for different extractors/configs
                (e.g., `"inception_v3"`, `"resnet50_pool5"`).
            force_recompute: If True, bypass cache and recompute, then overwrite
                the cache entry.

        Returns:
            np.ndarray: The feature matrix for all samples in `loader`.

        Logging:
            Emits INFO on cache hits/misses and DEBUG with the resolved cache path.
        """
        cache_key = self._generate_cache_key(loader, cache_key_suffix)
        logger.debug(
            f"Generated cache key: {cache_key} for loader with dataset: {type(loader.dataset).__name__}"
        )
        cache_path = self._get_cache_path(cache_key)
        logger.debug(f"Cache path resolved to: {cache_path}")

        if not force_recompute:
            cached_features = self._load_from_cache(cache_path)
            if cached_features is not None:
                logger.info(f"Loaded features from cache: {cache_path}")
                return cached_features

        if force_recompute:
            logger.info(
                f"Force recompute requested, computing features and saving to {cache_path}"
            )
        else:
            logger.info(f"Cache miss, computing features and saving to {cache_path}")

        features = compute_fn(loader)
        self._save_to_cache(features, cache_path)
        return features

    def _generate_cache_key(
        self, loader: DataLoader[VisionDataset], suffix: str = ""
    ) -> str:
        """Create a deterministic MD5 key from dataset identity and loader config.

        Key composition (as a stringified list) includes:
            - dataset identifier:
                * for `Subset`: underlying dataset root (if present) or
                  `<DatasetName>_<len>`, plus a hash of **sorted indices**;
                * for datasets with `.root`: absolute resolved path;
                * otherwise: `<DatasetName>_<len(dataset)>`.
            - `suffix`: feature-extractor/config disambiguator.
            - `len(dataset)`: dataset size.
            - `loader.batch_size`: to avoid mixing caches produced with
              different batching strategies.

        Args:
            loader: The DataLoader to inspect (dataset and params).
            suffix: Optional extractor/config suffix.

        Returns:
            str: MD5 hex digest used as the cache file stem.
        """
        dataset = loader.dataset

        # Handle Subset datasets (from subsampling) specially
        if isinstance(dataset, Subset):
            underlying_dataset = dataset.dataset
            indices = sorted(dataset.indices)  # Sort for consistency

            if (
                hasattr(underlying_dataset, "root")
                and underlying_dataset.root is not None
            ):
                dataset_id = str(Path(underlying_dataset.root).resolve())
            else:
                dataset_id = (
                    f"{type(underlying_dataset).__name__}_{len(underlying_dataset)}"
                )

            dataset_id = f"{dataset_id}_subset_{hash(tuple(indices))}"
            logger.debug(
                f"Using subset cache key with underlying dataset: {dataset_id}"
            )

        elif hasattr(dataset, "root") and dataset.root is not None:
            dataset_id = str(Path(dataset.root).resolve())
            logger.debug(f"Using dataset root for cache key: {dataset_id}")
        else:
            dataset_id = f"{type(dataset).__name__}_{len(dataset)}"
            logger.warning(
                f"Dataset does not have a 'root' attribute. Using fallback: {dataset_id}"
            )

        key_components = [
            dataset_id,
            suffix,              # Feature extractor type or variant
            len(dataset),        # Dataset size
            loader.batch_size,   # Batch size
        ]

        key_string = str(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Return the filesystem path to the cache file for `cache_key`."""
        return self.cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_path: Path) -> np.ndarray | None:
        """Load a cached feature matrix from disk, if present and valid.

        Args:
            cache_path: Path to the pickle file.

        Returns:
            np.ndarray | None: The loaded feature matrix, or `None` if the file
            is missing/corrupt and was removed.

        Notes:
            - On unpickling errors or truncated files, the cache file is deleted
              to avoid repeated failures.
        """
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                features = pickle.load(f)
            return features
        except (pickle.UnpicklingError, FileNotFoundError, EOFError) as e:
            print(f"Failed to load cache from {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def _save_to_cache(self, features: np.ndarray, cache_path: Path) -> None:
        """Persist a feature matrix to disk using highest pickle protocol.

        Args:
            features: Feature array to store (e.g., shape `(N, D)`).
            cache_path: Destination file path.

        Side effects:
            Writes/overwrites `cache_path`. On failure, removes any partial file.
        """
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Failed to save cache to {cache_path}: {e}")
            cache_path.unlink(missing_ok=True)
