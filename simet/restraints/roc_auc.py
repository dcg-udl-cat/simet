import logging
from typing import override

from simet.dataset_loaders import DatasetLoader
from simet.metrics import RocAuc
from simet.restraints import Restraint

logger = logging.getLogger(__name__)


class RocAucRestraint(Restraint[float]):
    """Restraint on the **ROC AUC** score computed from real vs. synthetic features.

    Wraps :class:`RocAuc` and checks that the resulting score falls within the
    inclusive interval ``[lower_bound, upper_bound]`` when those bounds are set.

    Semantics:
        - ROC AUC is in ``[0.0, 1.0]``; higher is generally better, with
          ~0.5 indicating chance level. Choose bounds accordingly.

    Requirements:
        - The provided :class:`DatasetLoader` must expose `real_features` and
          `synth_features` as 2D arrays with the same feature dimension.

    Args (via base class):
        lower_bound (float | None): Inclusive minimum ROC AUC. If `None`, no lower check.
        upper_bound (float | None): Inclusive maximum ROC AUC. If `None`, no upper check.

    Returns (from ``apply``):
        tuple[bool, float]: ``(passes, value)`` where ``value`` is the ROC AUC
        and ``passes`` indicates whether it lies within the configured bounds.
    """

    @override
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ROC AUC restraint and its underlying metric."""
        super().__init__(*args, **kwargs)
        self.metric = RocAuc()

    @override
    def apply(self, loader: DatasetLoader) -> tuple[bool, float]:
        """Compute ROC AUC and evaluate it against the configured bounds.

        Args:
            loader (DatasetLoader): Must provide compatible `real_features` and
                `synth_features` for the ROC AUC computation.

        Returns:
            tuple[bool, float]: ``(passes, roc_auc)`` where ``passes`` is True iff
            ``lower_bound <= roc_auc <= upper_bound`` (treating `None` as unbounded).
        """
        roc_auc = self.metric.compute(loader)
        lower_ok = self.lower_bound is None or roc_auc >= self.lower_bound
        upper_ok = self.upper_bound is None or roc_auc <= self.upper_bound
        passes = lower_ok and upper_ok
        logger.info(f"ROC AUC Restraint passes: {passes}")
        return passes, roc_auc
