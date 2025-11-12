from dataclasses import dataclass


@dataclass
class FeatureExtractorSchema:
    """Config schema for selecting a feature extractor (extensible).

    This schema intentionally keeps `type` as a free string so users can add
    new extractors without changing the schema—your factory/registry is
    responsible for resolving the string into a concrete implementation.

    Attributes:
        type (str):
            Identifier of the extractor backend (e.g., "inception_v3"). 
            The valid values are defined by your
            feature-extractor registry/factory at runtime.

    Example:
        >>> cfg = FeatureExtractorSchema(type="inception_v3")
        >>> cfg.type
        'inception_v3'

    Notes:
        - Add optional fields here (e.g., `cache_dir`, `force_recompute`,
          layer names, normalization presets) as your extractors evolve.
    """
    type: str
