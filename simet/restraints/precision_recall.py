import logging
from typing import override

from simet.dataset_loaders import DatasetLoader
from simet.metrics import PrecisionRecall
from simet.restraints import Restraint

logger = logging.getLogger(__name__)


class PrecisionRecallRestraint(Restraint[tuple[float, float]]):
    """Restraint on the `(precision, recall)` pair computed over features.

    Wraps :class:`PrecisionRecall` and checks that both precision and recall
    lie within the **inclusive**, element-wise interval:

        ``lower_bound <= (precision, recall) <= upper_bound``

    Requirements:
        - ``DatasetLoader`` must expose 2D `real_features` and `synth_features`
          with the same feature dimension.

    Args:
        lower_bound (tuple[float, float] | None, optional):
            Inclusive minimums for `(precision, recall)`. Defaults to `(0.0, 0.0)`.
            Use `None` for no lower constraint.
        upper_bound (tuple[float, float] | None, optional):
            Inclusive maximums for `(precision, recall)`. Defaults to `(1.0, 1.0)`.
            Use `None` for no upper constraint.

    Returns (from ``apply``):
        tuple[bool, tuple[float, float]]: ``(passes, (precision, recall))`` where
        ``passes`` is True iff both metrics fall within the configured bounds.

    Notes:
        - Bounds are applied **element-wise** (precision vs precision bounds,
          recall vs recall bounds).
        - Precision and recall are in `[0.0, 1.0]`.
    """

    @override
    def __init__(
        self, 
        lower_bound: tuple[float, float] | None = (0.0, 0.0),
        upper_bound: tuple[float, float] | None = (1.0, 1.0),
        ) -> None:
        """Initialize the restraint and its underlying metric."""
        super().__init__(lower_bound, upper_bound)
        self.metric = PrecisionRecall()

    @override
    def apply(self, loader: DatasetLoader) -> tuple[bool, tuple[float, float]]:
        """Compute precision/recall and evaluate them against the bounds.

        Args:
            loader (DatasetLoader): Source of `real_features` and `synth_features`.

        Returns:
            tuple[bool, tuple[float, float]]: ``(passes, (precision, recall))``.
        """
        precision, recall = self.metric.compute(loader)
        value = (precision, recall)
        passes = True

        # Upper bounds (inclusive)
        if self.upper_bound is not None:
            precision_upper, recall_upper = self.upper_bound
            if precision > precision_upper or recall > recall_upper:
                passes = False

        # Lower bounds (inclusive)
        if self.lower_bound is not None:
            precision_lower, recall_lower = self.lower_bound
            if precision < precision_lower or recall < recall_lower:
                passes = False

        logger.info(f"Precision/Recall Restraint passes: {passes}")
        return passes, value
