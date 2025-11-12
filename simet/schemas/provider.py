from dataclasses import dataclass


@dataclass
class ProviderSchema:
    """Config schema for selecting and configuring a data provider (extensible).

    Keeps `type` as a free string so users can register new providers without
    touching the schema. Your provider **factory/registry** resolves this name
    to a concrete implementation at runtime.

    Attributes:
        type (str):
            Identifier of the provider backend (e.g., "LocalProviderWithClass",
            "LocalProviderWithoutClass", "CIFARProvider", "MyCustomProvider").
            The accepted values are defined by your provider registry.
        path (str):
            Filesystem root used by the provider. For local providers, this is
            typically a directory on disk. For downloadable datasets, it may be
            a cache/root directory.

    Example:
        >>> ProviderSchema(type="LocalProviderWithClass", path="data/pets")
        >>> ProviderSchema(type="CIFARProvider", path="data/cifar10")

    Notes:
        - Consider expanding the schema with optional, provider-specific fields
          (e.g., split, recursive, class_map) as your ecosystem grows.
    """
    type: str
    path: str
