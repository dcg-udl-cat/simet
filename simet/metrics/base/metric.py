from abc import ABC, abstractmethod

from simet.dataset_loaders import DatasetLoader


class Metric[T](ABC):
    """Abstract interface for evaluation metrics computed from a `DatasetLoader`.

    This generic base class defines the minimal contract every metric must
    implement: a human-readable `name` and a `compute(loader)` method that
    returns a result of type `T` (e.g., `float`, `tuple[float, float]`, or a
    small dataclass).

    Type Parameters:
        T: The result type returned by :meth:`compute`. Examples: `float` for
            scalar scores (FID, ROC AUC) or `tuple[float, float]` for
            multi-valued metrics (precision, recall).

    Example:
        >>> class Accuracy(Metric[float]):
        ...     @property
        ...     def name(self) -> str:
        ...         return "Accuracy"
        ...
        ...     def compute(self, loader: DatasetLoader) -> float:
        ...         # compute accuracy from loader ...
        ...         return 0.95
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name (used in logs and reports)."""
        pass

    @abstractmethod
    def compute(self, loader: DatasetLoader) -> T:
        """Compute the metric using data/features from `loader`.

        Implementations may pull precomputed features from the loader or run
        their own forward passes as needed.

        Args:
            loader (DatasetLoader): Source of datasets/dataloaders and (optionally)
                precomputed feature arrays.

        Returns:
            T: The metric result (scalar or structured), as defined by the
            concrete metric implementation.
        """
        pass
