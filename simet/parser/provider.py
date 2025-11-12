import logging
from enum import StrEnum
from pathlib import Path

from simet.providers import (
    CIFARProvider,
    LocalProviderWithClass,
    LocalProviderWithoutClass,
    Provider,
)
from simet.schemas import ProviderSchema

logger = logging.getLogger(__name__)


class ProviderParser:
    """Factory wrapper that builds concrete providers from a config dict.

    Parses a `ProviderSchema` from a plain mapping (e.g., YAML/JSON) and
    delegates construction to :meth:`ProviderType.get_provider`.
    """

    @staticmethod
    def parse_provider(provider_data: dict) -> Provider:
        """Parse provider config and return a concrete `Provider`.

        Args:
            provider_data (dict):
                Mapping that must match `ProviderSchema` fields, e.g.:
                `{"type": "LocalProviderWithClass", "path": "data/pets"}`.

        Returns:
            Provider: An instance of the requested provider.

        Raises:
            TypeError: If `provider_data` is missing required fields for `ProviderSchema`.
            ValueError: If the `"type"` value is unknown/unsupported.

        Example:
            >>> cfg = {"type": "LocalProviderWithoutClass", "path": "data/unlabeled"}
            >>> p = ProviderParser.parse_provider(cfg)
            >>> p.__class__.__name__
            'LocalProviderWithoutClass'
        """
        provider_schema = ProviderSchema(**provider_data)
        return ProviderType.get_provider(provider_schema)


class ProviderType(StrEnum):
    """Enum of supported providers (string-valued)."""

    LOCALPROVIDERCLASS = "LocalProviderWithClass"
    LOCALPROVIDERNOCLASS = "LocalProviderWithoutClass"
    CIFAR = "CIFARProvider"

    @staticmethod
    def get_provider(provider_schema: ProviderSchema) -> Provider:
        """Construct a `Provider` from a validated `ProviderSchema`.

        Converts the `path` to a `Path` and instantiates the provider indicated
        by `provider_schema.type`.

        Args:
            provider_schema (ProviderSchema): Validated schema with `type` and `path`.

        Returns:
            Provider: Concrete provider instance:
                - `"LocalProviderWithClass"` → `LocalProviderWithClass(path)`
                - `"LocalProviderWithoutClass"` → `LocalProviderWithoutClass(path)`
                - `"CIFARProvider"` → `CIFARProvider(path)`

        Raises:
            ValueError: If `provider_schema.type` is not a recognized `ProviderType`.

        Notes:
            - Local providers expect `path` to exist on disk; validation occurs
              when their `get_data(...)` is called.
            - `CIFARProvider` will download the dataset into `path` if missing.
        """
        try:
            provider_type = ProviderType(provider_schema.type)
            path = Path(provider_schema.path)
            match provider_type:
                case ProviderType.LOCALPROVIDERCLASS:
                    return LocalProviderWithClass(path)
                case ProviderType.LOCALPROVIDERNOCLASS:
                    return LocalProviderWithoutClass(path)
                case ProviderType.CIFAR:
                    return CIFARProvider(path)
        except ValueError as e:
            logger.error(f"Unknown provider type: {provider_schema.type}")
            raise ValueError(f"Unknown provider type: {provider_schema.type}") from e
