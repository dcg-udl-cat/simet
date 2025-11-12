from dataclasses import dataclass

from simet.schemas.feature_extractor import FeatureExtractorSchema
from simet.schemas.provider import ProviderSchema
from simet.schemas.transform import TransformSchema


@dataclass
class LoaderSchema:
    """Schema describing how to build a `DatasetLoader` (providers + transforms + FE).

    Encapsulates the components needed to construct the data loading and
    feature-extraction pipeline: one provider for real data, one for synthetic,
    a transform used for provider/datalaoder preprocessing, and the feature
    extractor configuration.

    Attributes:
        real_provider (ProviderSchema):
            Configuration for the **real** dataset provider.
        synth_provider (ProviderSchema):
            Configuration for the **synthetic** dataset provider.
        provider_transform (TransformSchema):
            Transform applied when constructing the provider-backed datasets
            and their dataloaders (i.e., preprocessing seen by the feature extractor).
        feature_extractor (FeatureExtractorSchema):
            Feature extractor selection/config (e.g., `"inception_v3"`).

    Example:
        >>> cfg = LoaderSchema(
        ...     real_provider=ProviderSchema(type="local_binary", data_path="data/real"),
        ...     synth_provider=ProviderSchema(type="local_binary", data_path="data/synth"),
        ...     provider_transform=TransformSchema(type="inception"),
        ...     feature_extractor=FeatureExtractorSchema(type="inception_v3"),
        ... )
        >>> cfg.real_provider.type
        'local_binary'

    Notes:
        - `real_provider` and `synth_provider` should be **symmetric** (same structure),
          differing mainly in their data paths.
        - The `provider_transform` should match the expectations of the
          `feature_extractor` to avoid feature drift (e.g., input size/normalization).
    """
    real_provider: ProviderSchema
    synth_provider: ProviderSchema
    provider_transform: TransformSchema
    feature_extractor: FeatureExtractorSchema
