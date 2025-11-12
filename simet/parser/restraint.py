import logging
from enum import StrEnum

from simet.restraints import (
    FIDRestraint,
    PrecisionRecallRestraint,
    Restraint,
    RocAucRestraint,
)

logger = logging.getLogger(__name__)


class RestraintParser:
    """Factory wrapper that builds restraint objects from config dicts.

    Converts a user/config value (e.g., from YAML/JSON) into a
    :class:`RestraintType` and delegates construction to
    :meth:`RestraintType.get_restraint`.
    """

    @staticmethod
    def parse_restraint(restraint_data: dict) -> Restraint:
        """Parse config and return a concrete restraint instance.

        Args:
            restraint_data (dict):
                Mapping that must include the key `"type"` matching a
                :class:`RestraintType` value (e.g., `"FIDRestraint"`), plus
                optional `"upper_bound"` and `"lower_bound"` fields whose
                shape depends on the restraint:
                  - FID / RocAuc: `float | None`
                  - PrecisionRecall: `Sequence[float] | None` for each bound

        Returns:
            Restraint: An instance of the requested restraint class.

        Raises:
            KeyError: If `"type"` is missing.
            ValueError: If the `"type"` value is unknown/unsupported.
        """
        restraint_type = RestraintType(restraint_data["type"])
        return RestraintType.get_restraint(restraint_type, restraint_data)


class RestraintType(StrEnum):
    """Enum of supported restraint identifiers (string-valued)."""

    FID = "FIDRestraint"
    PRECISIONRECALL = "PrecisionRecallRestraint"
    ROCAUC = "RocAucRestraint"

    @staticmethod
    def get_restraint(restraint_type: "RestraintType", restraint_data: dict) -> Restraint:
        """Construct a restraint instance for the given enum value.

        Expects `restraint_data` to provide `upper_bound` and/or `lower_bound`
        as appropriate for the metric. For Precision/Recall, list-like bounds
        are coerced to tuples.

        Args:
            restraint_type (RestraintType): Enum indicating which restraint to build.
            restraint_data (dict): Source config (typically parsed from YAML/JSON).

        Returns:
            Restraint: A concrete restraint instance:
                - `FIDRestraint(upper_bound: float | None, lower_bound: float | None)`
                - `PrecisionRecallRestraint(upper_bound: tuple[float, ...] | None, lower_bound: tuple[float, ...] | None)`
                - `RocAucRestraint(upper_bound: float | None, lower_bound: float | None)`

        Raises:
            ValueError: If the `restraint_type` is unknown (logged and re-raised).

        Notes:
            - `upper_bound` / `lower_bound` may be absent or `None`. Accessing
              them directly (e.g., `restraint_data["upper_bound"]`) will raise
              `KeyError`; adapt if you plan to make them optional.
            - Ensure the order/length of Precision/Recall bounds match that
              restraint’s output semantics.
        """
        try:
            match restraint_type:
                case RestraintType.FID:
                    return FIDRestraint(
                        upper_bound=restraint_data["upper_bound"],
                        lower_bound=restraint_data["lower_bound"],
                    )
                case RestraintType.PRECISIONRECALL:
                    return PrecisionRecallRestraint(
                        upper_bound=None if restraint_data["upper_bound"] is None
                        else tuple(restraint_data["upper_bound"]),
                        lower_bound=None if restraint_data["lower_bound"] is None
                        else tuple(restraint_data["lower_bound"]),
                    )
                case RestraintType.ROCAUC:
                    return RocAucRestraint(
                        upper_bound=restraint_data["upper_bound"],
                        lower_bound=restraint_data["lower_bound"],
                    )
        except ValueError as e:
            logger.error(f"Unknown restraint type: {restraint_type}")
            raise ValueError(f"Unknown restraint type: {restraint_type}") from e
