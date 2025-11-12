from pathlib import Path

from typer import Argument, Option, Typer

from simet.dataset_loaders import DatasetLoader
from simet.pipeline import Pipeline, SimplePipeline
from simet.providers import LocalProviderWithoutClass
from simet.services.logging import LoggingService
from simet.services.seeding import SeedingService

DEFAULT_LOG_PATH = Path.home() / ".simet" / "logs"

app = Typer(no_args_is_help=True, name="simet")


@app.command(
    help="Run a simple version of the pipeline from two paths. Will use Inception as the feature extractor and run all metrics (without restraints). Assumes images are un-labeled (i.e. no class subdirectories). For further customization, refer to using the full Pipeline in main.py.",
    short_help="Run a simple version of the pipeline with no restraints.",
)
def simple(
    real_path: Path = Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=False,
        help="Path to directory containing real images",
    ),
    synth_path: Path = Argument(  # noqa: B008
        ...,
        exists=True,
        file_okay=False,
        help="Path to directory containing synthetic images",
    ),
    log_path: Path | None = Option(  # noqa: B008
        DEFAULT_LOG_PATH,
        exists=False,
        file_okay=False,
        help="Path to directory where logs will be stored. If not provided, defaults to `~/.simet/logs`",
    ),
):
    """Run a simple pipeline to evaluate synthetic images against real images.

    Calculates FID, Precision/Recall and ROC AUC metrics using Inception feature extractor.
    Assumes unlabeled images (no class subdirectories).

    Args:
        real_path (Path): Path to directory containing real images.
        synth_path (Path): Path to directory containing synthetic images.
        log_path (Path | None, optional): Path to directory where logs will be stored.
            If not provided, defaults to `~/.simet/logs`.
    """
    LoggingService.setup_logging(log_path)
    SeedingService.set_global_seed()
    SimplePipeline(
        loader=DatasetLoader(
            real_provider=LocalProviderWithoutClass(
                real_path,
            ),
            synth_provider=LocalProviderWithoutClass(
                synth_path,
            ),
        ),
    ).run()


@app.command(
    help="Run the full pipeline from a `yaml` config file. Refer to documentation for examples. Uses restraints",
    short_help="Run the full pipeline with restraints.",
)
def pipeline(
    config_path: Path = Argument(  # noqa: B008
        ...,
        exists=True,
        dir_okay=False,
        help="Path to pipeline configuration file in `yaml` format",
    ),
    log_path: Path | None = Option(  # noqa: B008
        DEFAULT_LOG_PATH,
        exists=False,
        file_okay=False,
        help="Path to directory where logs will be stored. If not provided, defaults to `~/.simet/logs`",
    ),
):
    """Run the complete pipeline from a YAML configuration file.

    Supports FID, Precision/Recall, and ROC AUC metrics with configurable restraints.
    Refer to documentation for configuration examples.

    Args:
        config_path (Path): Path to pipeline configuration file in `yaml` format.
        log_path (Path | None, optional): Path to directory where logs will be stored. If not provided, defaults to `~/.simet/logs`.
    """
    LoggingService.setup_logging(log_path)
    SeedingService.set_global_seed()
    Pipeline.from_yaml(config_path).run()


@app.command(
    help="Run a simple version of the pipeline from a `yaml` config file. Refer to documentation for examples. No restraints are used.",
    short_help="Run a simple version of the pipeline with no restraints.",
)
def simple_pipeline(
    config_path: Path = Argument(  # noqa: B008
        ...,
        exists=True,
        dir_okay=False,
        help="Path to simple pipeline configuration file in `yaml` format",
    ),
    log_path: Path | None = Option(  # noqa: B008
        DEFAULT_LOG_PATH,
        exists=False,
        file_okay=False,
        help="Path to directory where logs will be stored. If not provided, defaults to `~/.simet/logs`",
    ),
):
    """Run a simplified pipeline from a YAML configuration file.

    Supports FID, Precision/Recall, and ROC AUC metrics without restraints.
    Uses simplified configuration format.

    Args:
        config_path (Path): Path to simple pipeline configuration file in `yaml` format.
        log_path (Path | None, optional): Path to directory where logs will be stored. If not provided, defaults to `~/.simet/logs`.
    """
    LoggingService.setup_logging(log_path)
    SeedingService.set_global_seed()
    SimplePipeline.from_yaml(config_path).run()


if __name__ == "__main__":
    app()
