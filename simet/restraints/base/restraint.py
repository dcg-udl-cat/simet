from abc import ABC, abstractmethod

from simet.dataset_loaders import DatasetLoader
from simet.metrics import Metric


class Restraint[T](ABC):
    """Abstract gate/threshold around a metric result.

    A `Restraint[T]` wraps a `Metric[T]` and (optionally) enforces lower/upper
    bounds on the computed value. Subclasses define how to evaluate the metric
    and how to compare the result against the configured bounds.

    Type Parameters:
        T: The metric’s return type (e.g., `float`, `tuple[float, float]`...). 
            Bounds should be of the same shape/type as `T`.

    Attributes:
        metric (Metric[T]): The metric instance evaluated by the restraint.
        lower_bound (T | None): Optional minimum acceptable value(s).
        upper_bound (T | None): Optional maximum acceptable value(s).

    Bound semantics:
        - Concrete subclasses must define the comparison logic inside `apply`.
        - Common convention for scalars: pass if
          `lower_bound <= value <= upper_bound` (with `None` meaning unbounded).
        - For structured outputs (e.g., tuples), define element-wise or
          custom logic and document it in the subclass.

    Example:
        >>> class MaxFID(Restraint[float]):
        ...     def __init__(self, fid_metric: Metric[float], upper_bound: float):
        ...         super().__init__(lower_bound=None, upper_bound=upper_bound)
        ...         self.metric = fid_metric
        ...     def apply(self, loader: DatasetLoader) -> tuple[bool, float]:
        ...         value = self.metric.compute(loader)   # e.g., 37.2
        ...         ok = (self.upper_bound is None) or (value <= self.upper_bound)  # inclusive
        ...         return ok, value
    """

    metric: Metric[T]
    lower_bound: T | None
    upper_bound: T | None

    def __init__(
        self, lower_bound: T | None = None, upper_bound: T | None = None
    ) -> None:
        """Initialize a restraint with optional bounds.

        Args:
            lower_bound: Minimum acceptable value(s) for the metric result,
                or `None` for no lower constraint.
            upper_bound: Maximum acceptable value(s) for the metric result,
                or `None` for no upper constraint.
        """
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @abstractmethod
    def apply(self, loader: DatasetLoader) -> tuple[bool, T]:
        """Evaluate the metric and decide pass/fail against the bounds.

        Implementations should:
          1) Compute the metric: `value = self.metric.compute(loader)`.
          2) Compare `value` against `lower_bound` / `upper_bound`
             using the subclass’s semantics.
          3) Return `(passes, value)` where `passes` is `True` iff the
             restraint is satisfied.

        Args:
            loader: Source of datasets/features for the metric computation.

        Returns:
            tuple[bool, T]: `(passes, value)` where `value` is the computed metric
            result of type `T`.

        Notes:
            - Be explicit in subclasses about inclusivity (≤/≥ vs </>) and how
              multi-valued results are checked (element-wise, aggregate, etc.).
        """
        pass
