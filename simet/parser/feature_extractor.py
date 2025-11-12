import logging
from enum import StrEnum

from simet.feature_extractor import FeatureExtractor, InceptionFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractorParser:
    """Factory wrapper that builds feature extractors from config dicts.

    Converts a user/config value (e.g., from YAML/JSON) into a
    :class:`FeatureExtractorType` and delegates construction to
    :meth:`FeatureExtractorType.get_feature_extractor`.
    """

    @staticmethod
    def parse_feature_extractor(feature_extractor_data: dict) -> FeatureExtractor:
        """Parse config and return a concrete feature extractor instance.

        Args:
            feature_extractor_data (dict):
                Mapping that must include the key `"type"` with a value matching
                a :class:`FeatureExtractorType` (e.g., `"InceptionFeatureExtractor"`).

        Returns:
            FeatureExtractor: An instance of the requested extractor
            (currently :class:`InceptionFeatureExtractor`).

        Raises:
            KeyError: If `"type"` is missing from `feature_extractor_data`.
            ValueError: If the `"type"` value is unknown/unsupported.

        Example:
            >>> cfg = {"type": "InceptionFeatureExtractor"}
            >>> fe = FeatureExtractorParser.parse_feature_extractor(cfg)
            >>> fe.__class__.__name__
            'InceptionFeatureExtractor'
        """
        feature_extractor_type = FeatureExtractorType(feature_extractor_data["type"])
        return FeatureExtractorType.get_feature_extractor(feature_extractor_type)


class FeatureExtractorType(StrEnum):
    """Enum of supported feature extractor identifiers (string-valued)."""

    INCEPTION = "InceptionFeatureExtractor"

    @staticmethod
    def get_feature_extractor(feature_extractor_type: "FeatureExtractorType") -> FeatureExtractor:
        """Construct a feature extractor for the given enum value.

        Args:
            feature_extractor_type (FeatureExtractorType):
                Enum member indicating which extractor to instantiate.

        Returns:
            FeatureExtractor: Concrete extractor instance.

        Raises:
            ValueError: If the type is unknown (logged and re-raised).

        Notes:
            - Extend this `match` with additional cases as new extractors are added.
        """
        try:
            match feature_extractor_type:
                case FeatureExtractorType.INCEPTION:
                    return InceptionFeatureExtractor()
        except ValueError as e:
            logger.error(f"Unknown feature extractor type: {feature_extractor_type}")
            raise ValueError(
                f"Unknown feature extractor type: {feature_extractor_type}"
            ) from e
