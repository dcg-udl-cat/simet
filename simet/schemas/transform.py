from dataclasses import dataclass


@dataclass
class TransformSchema:
    """Config schema for selecting a data transform (extensible).

    Keeps `type` as a free string so users can plug in new transforms via a
    factory/registry without changing the schema. The transform name is later
    resolved to a concrete implementation that returns a
    `torchvision.transforms.Compose`.

    Attributes:
        type (str):
            Identifier of the transform to use (e.g., "inception",
            "my_custom_transform"). The accepted values are
            defined by your transform registry at runtime.

    Example:
        >>> TransformSchema(type="inception")
        >>> TransformSchema(type="my_custom_transform")

    Notes:
        - Ensure the chosen transform matches the expectations of your feature
          extractor/model (e.g., input size, normalization) to avoid feature drift.
        - If you later need per-transform options (e.g., image size, mean/std,
          augmentation flags), extend this schema with optional fields.
    """
    type: str
