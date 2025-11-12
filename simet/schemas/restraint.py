from dataclasses import dataclass


@dataclass
class RestraintSchema:
    """Config schema for a metric restraint (threshold/check), extensible via registry.

    Keeps `type` as a free string so users can register new restraints/metrics
    without changing the schema. Each concrete restraint interprets the metric
    and how `upper_bound`/`lower_bound` are applied.

    Attributes:
        type (str):
            Identifier of the restraint/metric (e.g., "FIDRestraint",
            "PrecisionRecallRestraint", "RocAucRestraint", "MyCustomRestraint").
            The valid values are defined by your restraint registry/factory.
        upper_bound (float | list[float] | None):
            Optional maximum acceptable value(s). Use a list when the metric
            returns multiple values (e.g., `[precision, recall]`). `None` means
            no upper constraint.
        lower_bound (float | list[float] | None):
            Optional minimum acceptable value(s). Use a list for multi-output
            metrics. `None` means no lower constraint.

    Examples:
        # Single-output metric (ROC AUC): require >= 0.85
        >>> RestraintSchema(type="RocAucRestraint", lower_bound=0.85)

        # Single-output metric (FID): require <= 40.0
        >>> RestraintSchema(type="FIDRestraint", upper_bound=40.0)

        # Multi-output metric (precision, recall): precision>=0.7, recall>=0.6
        >>> RestraintSchema(type="PrecisionRecallRestraint", lower_bound=[0.7, 0.6])

    Notes:
        - Ensure list bounds **length and order** match the metric’s output.
        - Inclusivity/exclusivity (≤ vs <, ≥ vs >) is defined by the concrete restraint.
        - You can extend this schema with metric-specific params later (e.g., `k`, `seed`).
    """
    type: str
    upper_bound: float | list[float] | None = None
    lower_bound: float | list[float] | None = None
