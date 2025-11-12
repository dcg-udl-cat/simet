import logging
from pathlib import Path

import yaml

from simet.dataset_loaders import DatasetLoader
from simet.metrics import FID, Metric, PrecisionRecall, RocAuc
from simet.parser import SimpleMetricParser, SimpleProviderParser

logger = logging.getLogger(__name__)

PIPELINE_KEY = "pipeline"
REAL_PROVIDER_KEY = "real_path"
SYNTH_PROVIDER_KEY = "synth_path"
METRICS_KEY = "metrics"


class SimplePipeline:
    """Minimal pipeline that loads two folders and computes selected metrics.

    This pipeline wires a :class:`DatasetLoader` with two local providers
    (parsed from filesystem paths) and computes one or more metrics. If no
    metrics are specified, it defaults to running **FID**, **RocAuc**, and
    **PrecisionRecall** in that order.

    Expected YAML structure for :meth:`from_yaml`:

    ```yaml
    pipeline:
      real_path: data/real_images
      synth_path: data/synth_images
      metrics:        # optional; defaults to ["FID", "RocAuc", "PrecisionRecall"]
        - FID
        - RocAuc
        - PrecisionRecall
    ```

    Attributes:
        loader (DatasetLoader): Loader built from the two provided paths.
        metrics (list[Metric]): Metrics to compute; defaults to [FID, RocAuc, PrecisionRecall].

    Example:
        >>> sp = SimplePipeline.from_yaml(Path("simple.yaml"))
        >>> sp.run()  # computes each metric and logs the results
    """

    metrics: list[Metric]
    loader: DatasetLoader

    def __init__(
        self, loader: DatasetLoader, metrics: list[Metric] | None = None
    ) -> None:
        """Initialize the simple pipeline.

        Args:
            loader: Prepared dataset loader (constructed from two providers).
            metrics: Optional list of metric instances. If ``None``, falls back to
                ``[FID(), RocAuc(), PrecisionRecall()]``.
        """
        self.loader = loader
        self.metrics = metrics or [
            FID(),
            RocAuc(),
            PrecisionRecall(),
        ]

    def run(self) -> None:
        """Compute all configured metrics sequentially.

        For each metric:
          1) Logs the metric name.
          2) Calls ``metric.compute(self.loader)``.
          3) Logs the computed result.

        Returns:
            None
        """
        for metric in self.metrics:
            logger.info(f"Computing metric: {metric.name}")
            res = metric.compute(self.loader)
            logger.info(f"Metric {metric.name} computed successfully, result is {res}.")

        logger.info("All metrics computed successfully")

    @classmethod
    def from_yaml(cls, config_path: Path) -> "SimplePipeline":
        """Build a simple pipeline from a YAML file.

        Args:
            config_path: Path to a YAML file with the structure shown in the class docstring.

        Returns:
            SimplePipeline: A pipeline instance ready to run.

        Raises:
            OSError: If the file cannot be opened.
            yaml.YAMLError: If the YAML is invalid.
            ValueError: If required keys are missing (re-raised from `_from_config_dict`).
        """
        try:
            with open(config_path, "r") as file:
                pipeline_data = yaml.safe_load(file)
                return cls._from_config_dict(pipeline_data)
        except Exception as e:
            logger.error(f"Failed to parse pipeline file: {e}")
            raise

    @classmethod
    def _from_config_dict(cls, pipeline_data: dict) -> "SimplePipeline":
        """Build a simple pipeline from an in-memory mapping.

        Required keys:
            - ``pipeline.real_path``: Filesystem path to **real** images.
            - ``pipeline.synth_path``: Filesystem path to **synthetic** images.

        Optional keys:
            - ``pipeline.metrics``: List of metric names (strings) parsed by
              :class:`SimpleMetricParser` (e.g., ``["FID", "RocAuc"]``).

        Args:
            pipeline_data: Parsed YAML/JSON configuration.

        Returns:
            SimplePipeline: Pipeline with loader and selected metrics.

        Raises:
            ValueError: If any required key is missing.
        """
        metrics = cls._build_metrics(pipeline_data)
        try:
            return SimplePipeline(
                loader=DatasetLoader(
                    real_provider=SimpleProviderParser.parse_provider(
                        pipeline_data[PIPELINE_KEY][REAL_PROVIDER_KEY]
                    ),
                    synth_provider=SimpleProviderParser.parse_provider(
                        pipeline_data[PIPELINE_KEY][SYNTH_PROVIDER_KEY]
                    ),
                ),
                metrics=metrics,
            )
        except KeyError as e:
            logger.error(f"Missing required pipeline configuration key: {e}")
            raise ValueError(f"Missing required pipeline configuration key: {e}") from e

    @classmethod
    def _build_metrics(cls, pipeline_data: dict) -> list[Metric] | None:
        """Parse the metrics list from config or fall back to defaults.

        Args:
            pipeline_data: Parsed YAML/JSON configuration.

        Returns:
            list[Metric] | None: List of metric instances built by
            :class:`SimpleMetricParser`, or ``None`` to signal the default set.

        Notes:
            - If the ``metrics`` key is absent, a warning is logged and the
              default metric list is used.
        """
        try:
            metrics_data = pipeline_data[PIPELINE_KEY][METRICS_KEY]
            return [SimpleMetricParser.parse_metric(metric) for metric in metrics_data]
        except KeyError:
            logger.warning(
                "Haven't detected metrics in the config file. Falling back to default and applying all metrics"
            )
            return None
