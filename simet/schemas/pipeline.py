from dataclasses import dataclass

from simet.schemas.loader import LoaderSchema
from simet.schemas.restraint import RestraintSchema


@dataclass
class PipelineSchema:
    """Schema describing a full evaluation pipeline configuration.

    Combines the **loader** section (providers, transforms, feature extractor)
    with an ordered list of **restraints** (checks/metrics/validators) to run.

    Attributes:
        loader (LoaderSchema):
            Configuration for building datasets, dataloaders, and the feature extractor.
        restraints (list[RestraintSchema]):
            Ordered list of pipeline steps to apply after feature extraction.
            Each restraint defines its own type and parameters.

    Example:
        >>> cfg = PipelineSchema(
        ...     loader=LoaderSchema(...),
        ...     restraints=[
        ...         RestraintSchema(type="fid"),
        ...         RestraintSchema(type="precision_recall", k=5),
        ...     ],
        ... )
        >>> len(cfg.restraints) >= 1
        True

    Notes:
        - Execution order matters: restraints are applied **in sequence**.
        - Keep the loader’s transform aligned with the feature extractor’s expectations.
    """
    loader: LoaderSchema
    restraints: list[RestraintSchema]
