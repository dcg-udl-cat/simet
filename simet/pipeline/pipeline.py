import logging
from pathlib import Path

import yaml

from simet.dataset_loaders import DatasetLoader
from simet.parser import (
    FeatureExtractorParser,
    ProviderParser,
    RestraintParser,
    TransformParser,
)
from simet.restraints import Restraint

PIPELINE_KEY = "pipeline"
LOADER_KEY = "loader"
REAL_PROVIDER_KEY = "real_provider"
SYNTH_PROVIDER_KEY = "synth_provider"
PROVIDER_TRANSFORM_KEY = "provider_transform"
FEATURE_EXTRACTOR_KEY = "feature_extractor"
RESTRAINTS_KEY = "restraints"

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates dataset loading, feature extraction, and restraint checks.

    The `Pipeline` wires together a :class:`DatasetLoader` (providers, transform,
    feature extractor) and an ordered list of :class:`Restraint` instances.
    Calling :meth:`run` applies each restraint in sequence and short-circuits on
    the first failure.

    Attributes:
        loader (DatasetLoader): Constructed data/feature loader.
        restraints (list[Restraint]): Ordered list of checks/metrics to apply.

    Example:
        Minimal YAML structure expected by :meth:`from_yaml`:

        ```yaml
        pipeline:
          loader:
            real_provider:
              type: LocalProviderWithClass
              path: data/real
            synth_provider:
              type: LocalProviderWithClass
              path: data/synth
            provider_transform:
              type: InceptionTransform
            feature_extractor:
              type: InceptionFeatureExtractor
          restraints:
            - type: FIDRestraint
              upper_bound: 40.0
            - type: RocAucRestraint
              lower_bound: 0.85
        ```

        >>> p = Pipeline.from_yaml(Path("pipeline.yaml"))
        >>> ok = p.run()  # returns True iff all restraints pass
    """

    restraints: list[Restraint]
    loader: DatasetLoader

    def __init__(self, loader: DatasetLoader, restraints: list[Restraint]) -> None:
        """Create a pipeline from a loader and a list of restraints.

        Args:
            loader: Prepared :class:`DatasetLoader` (providers, transforms, FE).
            restraints: Ordered list of :class:`Restraint` instances to evaluate.
        """
        self.loader = loader
        self.restraints = restraints

    def run(self) -> bool:
        """Execute restraints in order; stop at first failure.

        Iterates over `self.restraints`, logs the metric name, and calls
        `restraint.apply(self.loader)`. If any check fails (`passes is False`),
        logs a warning and returns `False`. Returns `True` only if all pass.

        Returns:
            bool: `True` if all restraints pass; `False` otherwise.
        """
        for restraint in self.restraints:
            logger.info(f"Applying restraint: {restraint.metric.name}")
            passes, _ = restraint.apply(self.loader)
            if not passes:
                logger.warning(
                    f"Restraint {restraint.metric.name} failed. Stopping pipeline."
                )
                return False

        logger.info("All restraints passed successfully")
        return True

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Pipeline":
        """Construct a pipeline from a YAML file on disk.

        Parses the YAML located at `config_path`, validates required sections,
        and delegates to :meth:`_from_config_dict`.

        Args:
            config_path (Path): Path to a YAML file matching the expected schema.

        Returns:
            Pipeline: A ready-to-run pipeline instance.

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
    def _from_config_dict(cls, pipeline_data: dict) -> "Pipeline":
        """Construct a pipeline from an in-memory config mapping.

        Expects the following nested keys (see module constants):
            - `pipeline.loader.real_provider`
            - `pipeline.loader.synth_provider`
            - `pipeline.loader.provider_transform`
            - `pipeline.loader.feature_extractor`
            - `pipeline.restraints` (list)

        The method uses parser factories to turn the dicts into concrete objects:
        `ProviderParser`, `TransformParser`, `FeatureExtractorParser`,
        and `RestraintParser`.

        Args:
            pipeline_data (dict): Parsed YAML/JSON configuration.

        Returns:
            Pipeline: A pipeline assembled from the provided configuration.

        Raises:
            ValueError: If any required key is missing (logged and re-raised).
        """
        try:
            return cls(
                loader=DatasetLoader(
                    real_provider=ProviderParser.parse_provider(
                        pipeline_data[PIPELINE_KEY][LOADER_KEY][REAL_PROVIDER_KEY]
                    ),
                    synth_provider=ProviderParser.parse_provider(
                        pipeline_data[PIPELINE_KEY][LOADER_KEY][SYNTH_PROVIDER_KEY]
                    ),
                    provider_transform=TransformParser.parse_transform(
                        pipeline_data[PIPELINE_KEY][LOADER_KEY][PROVIDER_TRANSFORM_KEY]
                    ),
                    feature_extractor=FeatureExtractorParser.parse_feature_extractor(
                        pipeline_data[PIPELINE_KEY][LOADER_KEY][FEATURE_EXTRACTOR_KEY]
                    ),
                ),
                restraints=[
                    RestraintParser.parse_restraint(restraint)
                    for restraint in pipeline_data[PIPELINE_KEY][RESTRAINTS_KEY]
                ],
            )
        except KeyError as e:
            logger.error(f"Missing required pipeline configuration key: {e}")
            raise ValueError(f"Missing required pipeline configuration key: {e}") from e
