from typing import override

from simet.dataset_loaders import DatasetLoader
from simet.metrics import SampleDownstreamTask


class SampleTRTS(SampleDownstreamTask):
    """Toy downstream task using **T**rain on **R**eal, **T**est on **S**ynth splits.

    Trains the underlying model on `loader.real_downstream_dataloader` and
    evaluates on `loader.synth_downstream_dataloader`, returning the final
    **accuracy** reported by the parent implementation.

    Requirements:
        - `DatasetLoader` must have been constructed with a `downstream_transform`
          so that `real_downstream_dataloader` and `synth_downstream_dataloader`
          are available. Otherwise, an `AttributeError` will be raised when
          accessing these attributes.

    Example:
        >>> task = SampleTRTS()
        >>> score = task.compute(loader)  # uses real for train, synth for test
        >>> 0.0 <= score <= 1.0
        True
    """

    @override
    @property
    def name(self) -> str:
        """Human-readable task name."""
        return "Sample - TRTS"

    @override
    def compute(self, loader: DatasetLoader) -> float:
        """Train on real, test on synth, and return accuracy.

        Args:
            loader (DatasetLoader): Must expose
                `real_downstream_dataloader` and `synth_downstream_dataloader`
                (present only if a `downstream_transform` was provided).

        Returns:
            float: Final test accuracy on the synthetic set in `[0.0, 1.0]`.
        """
        return self._compute(
            train_set=loader.real_downstream_dataloader,
            test_set=loader.synth_downstream_dataloader,
        )
