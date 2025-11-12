import logging
from enum import StrEnum

from simet.schemas import TransformSchema
from simet.transforms import InceptionTransform, Transform

logger = logging.getLogger(__name__)


class TransformParser:
    """Factory that builds transform objects from config dicts.

    Parses a plain mapping (e.g., from YAML/JSON) into a `TransformSchema` and
    delegates construction to :meth:`TransformType.get_transform`.
    """

    @staticmethod
    def parse_transform(transform_data: dict) -> Transform:
        """Parse transform config and return a concrete transform instance.

        Args:
            transform_data (dict):
                Mapping that must match `TransformSchema` fields, e.g.:
                `{"type": "InceptionTransform"}`.

        Returns:
            Transform: An instance of the requested transform
            (currently `InceptionTransform`).

        Raises:
            TypeError: If required fields for `TransformSchema` are missing.
            ValueError: If the `"type"` value is unknown.

        Example:
            >>> cfg = {"type": "InceptionTransform"}
            >>> t = TransformParser.parse_transform(cfg)
            >>> t.__class__.__name__
            'InceptionTransform'
        """
        transform_schema = TransformSchema(**transform_data)
        return TransformType.get_transform(transform_schema)


class TransformType(StrEnum):
    """Enum of supported transform identifiers (string-valued)."""

    INCEPTION = "InceptionTransform"

    @staticmethod
    def get_transform(transform_schema: TransformSchema) -> Transform:
        """Construct a transform instance from a validated schema.

        Args:
            transform_schema (TransformSchema): Validated schema with a `type` name.

        Returns:
            Transform: Concrete transform instance.

        Raises:
            ValueError: If `transform_schema.type` is not a recognized `TransformType`.

        Notes:
            - Extend the `match` with additional cases as new transforms are added.
        """
        try:
            transform_type = TransformType(transform_schema.type)
            match transform_type:
                case TransformType.INCEPTION:
                    return InceptionTransform()
        except ValueError as e:
            logger.error(f"Unknown transform type: {transform_schema.type}")
            raise ValueError(f"Unknown transform type: {transform_schema.type}") from e
