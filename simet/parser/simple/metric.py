import logging
from enum import StrEnum

from simet.metrics import FID, Metric, PrecisionRecall, RocAuc

logger = logging.getLogger(__name__)


class SimpleMetricParser:
    """Factory wrapper that instantiates metric objects from strings.

    Converts a user-provided metric name into a `MetricType` enum and delegates
    to `MetricType.get_metric(...)` to construct the corresponding metric
    instance.

    Example:
        >>> SimpleMetricParser.parse_metric("FID").__class__.__name__
        'FID'
        >>> SimpleMetricParser.parse_metric("PrecisionRecall").__class__.__name__
        'PrecisionRecall'
    """

    @staticmethod
    def parse_metric(metric: str) -> Metric:
        """Parse a metric name and return a concrete metric instance.

        Args:
            metric (str): Case-sensitive name matching `MetricType` values,
                e.g., `"FID"`, `"PrecisionRecall"`, or `"RocAuc"`.

        Returns:
            Metric: An instance of the corresponding metric class
            (e.g., `FID()`, `PrecisionRecall()`, `RocAuc()`).

        Raises:
            ValueError: If `metric` does not correspond to a known `MetricType`.
        """
        metric_type = MetricType(metric)
        return MetricType.get_metric(metric_type)


class MetricType(StrEnum):
    """Enum of supported metric identifiers (string-valued).

    Values:
        FID: Frechet Inception Distance.
        PRECISIONRECALL: Precision/Recall in feature space.
        ROCAUC: ROC AUC score.

    Notes:
        - `StrEnum` (Python 3.11+) ensures enum values are strings, so they can
          be compared directly to input strings.
    """

    FID = "FID"
    PRECISIONRECALL = "PrecisionRecall"
    ROCAUC = "RocAuc"

    @staticmethod
    def get_metric(metric_type: "MetricType") -> Metric:
        """Construct a metric instance for the given `MetricType`.

        Args:
            metric_type (MetricType): Enum value indicating which metric to build.

        Returns:
            Metric: An instance of the corresponding metric implementation.

        Raises:
            ValueError: If the metric type is unknown (logged and re-raised).
        """
        try:
            match metric_type:
                case MetricType.FID:
                    return FID()
                case MetricType.PRECISIONRECALL:
                    return PrecisionRecall()
                case MetricType.ROCAUC:
                    return RocAuc()
        except ValueError as e:
            logger.error(f"Unknown metric type: {metric_type}")
            raise ValueError(f"Unknown metric type: {metric_type}") from e
