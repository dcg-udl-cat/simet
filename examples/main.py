from pathlib import Path

from simet.dataset_loaders import DatasetLoader
from simet.feature_extractor import InceptionFeatureExtractor
from simet.pipeline import Pipeline
from simet.providers import LocalBinaryProvider
from simet.restraints import (
    FIDRestraint,
    PrecisionRecallRestraint,
    RocAucRestraint,
    SampleTRTSRestraint,
    SampleTSTRRestraint,
)
from simet.services import LoggingService, SeedingService
from simet.transforms import InceptionTransform, SampleDownstreamTransform


def main():
    # Logging & seeding
    LoggingService.setup_logging(  # Change this in the example
        log_dir=Path("tmp_dir")
    )
    SeedingService.set_global_seed()

    # Pipeline
    pipeline = Pipeline(
        loader=DatasetLoader(
            real_provider=LocalBinaryProvider(
                Path("data/real"),
            ),
            synth_provider=LocalBinaryProvider(
                Path("data/synth"),
            ),
            provider_transform=InceptionTransform(),
            feature_extractor=InceptionFeatureExtractor(),
            downstream_transform=SampleDownstreamTransform()
        ),
        restraints=[
            FIDRestraint(
                upper_bound=30.0, 
                lower_bound=0.0
            ),
            PrecisionRecallRestraint(
                upper_bound=(1.0, 1.0), # Precision upper bound, Recall upper bound
                lower_bound=(0.7, 0.7), # Precision lower bound, Recall lower bound
            ),
            RocAucRestraint(
                upper_bound=0.6, 
                lower_bound=0.4
            ),
            SampleTRTSRestraint(
                upper_bound=0.95, 
                lower_bound=0.0,
            ),
            SampleTSTRRestraint(
                upper_bound=0.95,
                lower_bound=0.0,
            ),
        ]
    )
    pipeline.run()

if __name__ == "__main__":
    main()
