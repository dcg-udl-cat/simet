import logging
from typing import override

from simet.dataset_loaders import DatasetLoader
from simet.metrics import FID
from simet.restraints import Restraint

logger = logging.getLogger(__name__)


class FIDRestraint(Restraint[float]):
    """Restraint on the **FID** score computed from real vs. synthetic features.

    Wraps :class:`FID` and checks that the resulting score falls within the
    inclusive interval ``[lower_bound, upper_bound]`` when those bounds are set.

    Semantics:
        - Lower FID is typically better. By default this restraint accepts
          values in ``[0.0, 500.0]``. Adjust bounds per your quality targets.

    Requirements:
        - The provided :class:`DatasetLoader` must expose `real_features` and
          `synth_features` as 2D arrays with matching feature dimension.

    Args:
        lower_bound (float | None, optional):
            Minimum acceptable FID (inclusive). Defaults to ``0.0``.
        upper_bound (float | None, optional):
            Maximum acceptable FID (inclusive). Defaults to ``500.0``.

    Returns (from ``apply``):
        tuple[bool, float]: ``(passes, value)`` where ``value`` is the FID
        (non-negative) and ``passes`` indicates whether it lies within bounds.
    """

    @override
    def __init__(
        self,
        lower_bound: float | None = 0.0,
        upper_bound: float | None = 500.0,
    ) -> None:
        """Initialize the FID restraint and its underlying metric."""
        super().__init__(lower_bound, upper_bound)
        self.metric = FID()

    @override
    def apply(self, loader: DatasetLoader) -> tuple[bool, float]:
        """Compute FID and evaluate it against the configured bounds.

        Args:
            loader (DatasetLoader): Must provide `real_features` and `synth_features`
                suitable for the FID computation.

        Returns:
            tuple[bool, float]: ``(passes, fid)`` where ``passes`` is True iff
            ``lower_bound <= fid <= upper_bound`` (treating `None` as unbounded).
        """
        fid = self.metric.compute(loader)
        lower_ok = self.lower_bound is None or fid >= self.lower_bound
        upper_ok = self.upper_bound is None or fid <= self.upper_bound
        passes = lower_ok and upper_ok
        logger.info(f"FID Restraint passes: {passes}")
        return passes, fid
