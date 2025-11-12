import logging

import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from simet.feature_extractor import FeatureExtractor, InceptionFeatureExtractor
from simet.providers import Provider
from simet.services import SubsamplingService
from simet.transforms import InceptionTransform, Transform

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and preprocess real vs. synthetic datasets, and compute features.

    Provides a unified interface to:
      1) subsample providers consistently,
      2) build `DataLoader`s with a given `Transform`, and
      3) extract feature arrays for downstream evaluation (e.g., FID, Precision/Recall, ROC-AUC).

    Args:
        real_provider (Provider):
            Provider that yields the **real** dataset (e.g., images). Must implement
            `get_data(transform)` returning a `torch.utils.data.Dataset`.
        synth_provider (Provider):
            Provider that yields the **synthetic** dataset, same interface as `real_provider`.
        provider_transform (Transform | None):
            Transform applied to both providers **for feature extraction**. If not provided,
            defaults to :class:`InceptionTransform`. This transform controls what the feature
            extractor will “see” (e.g., resizing/normalization for Inception).
        feature_extractor (FeatureExtractor | None):
            Feature extractor used to compute `real_features` and `synth_features`.
            Defaults to :class:`InceptionFeatureExtractor`.
        downstream_transform (Transform | None):
            Optional transform for **downstream tasks** (separate from feature extraction).
            When given, the loader also builds `real_downstream_dataloader` and
            `synth_downstream_dataloader` using this transform.

    Attributes:
        provider_transform (Transform):
            The transform actually used for feature extraction (defaults to `InceptionTransform` if None).
        real_dataloader (DataLoader[VisionDataset]):
            Dataloader for the real dataset with `provider_transform`.
        synth_dataloader (DataLoader[VisionDataset]):
            Dataloader for the synthetic dataset with `provider_transform`.
        real_features (np.ndarray):
            Feature matrix extracted from `real_dataloader`. Shape typically `(N_real, D)`.
        synth_features (np.ndarray):
            Feature matrix extracted from `synth_dataloader`. Shape typically `(N_synth, D)`.
        real_downstream_dataloader (DataLoader[VisionDataset]):
            Present only if `downstream_transform` is provided; uses that transform.
        synth_downstream_dataloader (DataLoader[VisionDataset]):
            Present only if `downstream_transform` is provided; uses that transform.

    Notes:
        - Providers are first **subsampled consistently** via `SubsamplingService.subsample(...)`
          using `provider_transform`, then converted into `DataLoader`s.
        - Feature extraction is performed eagerly at construction time by `feature_extractor`.
        - Private helpers (e.g., `_to_dataloader`, `_compute_features`) are internal and not
          part of the public API.
    """
    provider_transform: Transform
    real_dataloader: DataLoader[VisionDataset]
    real_features: np.ndarray
    synth_dataloader: DataLoader[VisionDataset]
    synth_features: np.ndarray

    # Downstream task specifics
    real_downstream_dataloader: DataLoader[VisionDataset]
    synth_downstream_dataloader: DataLoader[VisionDataset]

    def __init__(
        self,
        real_provider: Provider,
        synth_provider: Provider,
        provider_transform: Transform | None = None,
        feature_extractor: FeatureExtractor | None = None,
        downstream_transform: Transform | None = None,
    ) -> None:
        """
        Initialize the DatasetLoader, see class documentation for details.
        """
        self.provider_transform = provider_transform or InceptionTransform()
        real_provider, synth_provider = SubsamplingService.subsample(
            real_provider, synth_provider, self.provider_transform
        )
        self.real_dataloader = self._to_dataloader(real_provider, self.provider_transform)
        self.synth_dataloader = self._to_dataloader(synth_provider, self.provider_transform)
        feature_extractor = feature_extractor or InceptionFeatureExtractor()
        self._compute_features(feature_extractor)

        # Downstream task specifics
        if downstream_transform:
            self.real_downstream_dataloader = self._to_dataloader(real_provider, downstream_transform)
            self.synth_downstream_dataloader = self._to_dataloader(synth_provider, downstream_transform)


    def _to_dataloader(
        self,
        provider: Provider,
        transform: Transform,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> DataLoader[VisionDataset]:
        """Create a DataLoader for a given provider and transform.

        Args:
            provider (Provider): The data provider.
            transform (Transform): The transform to apply to the data.
            batch_size (int, optional): The batch size for the DataLoader. Defaults to 128.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            num_workers (int, optional): The number of worker processes to use. Defaults to 0.
            pin_memory (bool, optional): Whether to pin memory for the DataLoader. Defaults to True.

        Returns:
            DataLoader[VisionDataset]: The created DataLoader.
        """
        logger.info("Creating DataLoader for provider")
        return DataLoader(
            provider.get_data(transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )  # type: ignore

    def _compute_features(self, feature_extractor: FeatureExtractor) -> None:
        """Extract features from the datasets using the provided feature extractor.

        Args:
            feature_extractor (FeatureExtractor): The feature extractor to use.
        """
        logger.info("Extracting features for datasets")
        self.real_features = feature_extractor.extract_features(self.real_dataloader)
        self.synth_features = feature_extractor.extract_features(self.synth_dataloader)
