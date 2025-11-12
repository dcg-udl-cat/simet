import logging
from typing import override

from simet.dataset_loaders import DatasetLoader
from simet.metrics import SampleTSTR
from simet.restraints import Restraint

logger = logging.getLogger(__name__)


class SampleTSTRRestraint(Restraint[float]):
    """Restraint on the **SampleTSTR** downstream score (Train Synth → Test Real).

    Wraps :class:`SampleTSTR` and checks that the resulting **accuracy** lies
    within the inclusive interval ``[lower_bound, upper_bound]`` when those
    bounds are provided.

    Requirements:
        - The provided :class:`DatasetLoader` must have been built with a
          ``downstream_transform`` so that
          ``synth_downstream_dataloader`` and ``real_downstream_dataloader``
          are available to the underlying task.

    Args:
        lower_bound (float | None, optional): Minimum acceptable accuracy
            (inclusive). Defaults to 0.0.
        upper_bound (float | None, optional): Maximum acceptable accuracy
            (inclusive). Defaults to 1.0.

    Returns (from ``apply``):
        tuple[bool, float]: ``(passes, value)`` where ``value`` is the TSTR
        accuracy in ``[0.0, 1.0]`` and ``passes`` indicates whether the value
        is within the configured bounds.

    Example:
        >>> r = SampleTSTRRestraint(lower_bound=0.70)
        >>> passed, score = r.apply(loader)
        >>> passed and score >= 0.70
        True
    """

    @override
    def __init__(
        self, 
        lower_bound: float | None = 0.0,
        upper_bound: float | None = 1.0,
        ) -> None:
        """Initialize the TSTR restraint and its underlying metric."""
        super().__init__(lower_bound, upper_bound)
        # Propagate bounds explicitly (since we're not passing them to super()).
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.metric = SampleTSTR()

    @override
    def apply(self, loader: DatasetLoader) -> tuple[bool, float]:
        """Compute TSTR accuracy and evaluate it against the bounds.

        Args:
            loader: Dataset context with downstream dataloaders present.

        Returns:
            tuple[bool, float]: ``(passes, accuracy)`` where ``passes`` is True
            iff the accuracy is within the inclusive bounds.
        """
        fid = self.metric.compute(loader)
        lower_ok = self.lower_bound is None or fid >= self.lower_bound
        upper_ok = self.upper_bound is None or fid <= self.upper_bound
        passes = lower_ok and upper_ok
        logger.info(f"SampleTSTR restraint passes: {passes}")
        return passes, fid
