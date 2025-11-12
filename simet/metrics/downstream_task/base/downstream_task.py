import logging
from abc import ABC, abstractmethod

from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset

from simet.metrics import Metric

logger = logging.getLogger(__name__)


class DownstreamTask(Metric[float], ABC):
    """Abstract base for downstream evaluation tasks producing a scalar score.

    A `DownstreamTask` is a metric that **trains/evaluates a model** (or similar
    procedure) on a train split and reports a single scalar on a test split.
    It extends `Metric[float]`, so concrete subclasses must still 
    provide the `name` property and the public `compute(loader)` method 
    (unless implemented in a shared parent).

    Subclassing:
        Implement `_compute(train_set, test_set)` to perform the end-to-end
        training/evaluation and return a scalar. Manage devices, seeds, and
        early-stopping as needed inside this method.

    Expected data:
        - `train_set` and `test_set` are `DataLoader[VisionDataset]` instances
          that yield `(image_tensor, target)` pairs. Targets should be compatible
          with the task (e.g., `long` for classification).

    Example:
        >>> class LogisticRegressionROC(DownstreamTask):
        ...     @property
        ...     def name(self) -> str:
        ...         return "LogReg ROC-AUC"
        ...
        ...     def compute(self, loader) -> float:
        ...         # Split or obtain train/test loaders from `loader` as needed,
        ...         # then delegate to `_compute`.
        ...         return self._compute(loader.real_downstream_dataloader,
        ...                              loader.synth_downstream_dataloader)
        ...
        ...     def _compute(self, train_set, test_set) -> float:
        ...         # Fit linear head on features from train_set, evaluate ROC-AUC on test_set...
        ...         return 0.91
    """

    @abstractmethod
    def _compute(
        self, train_set: DataLoader[VisionDataset], test_set: DataLoader[VisionDataset]
    ) -> float:
        """Train/evaluate the downstream task on `train_set`/`test_set`.

        Args:
            train_set (DataLoader[VisionDataset]): Dataloader for training. Yields
                `(input, target)` pairs appropriate for the task.
            test_set (DataLoader[VisionDataset]): Dataloader for evaluation.

        Returns:
            float: Scalar performance score for the task (e.g., accuracy, ROC-AUC).

        Notes:
            - Handle device placement, mixed precision, and evaluation mode as needed.
            - For reproducibility, consider seeding (see `SeedingService`).
            - Clearly document in subclasses whether **higher is better**.
        """
        pass
