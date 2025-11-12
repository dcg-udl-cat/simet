import logging
import random

from torch.utils.data import Subset

from simet.providers import Provider, SubsampledProvider
from simet.transforms import Transform

logger = logging.getLogger(__name__)


class SubsamplingService:
    """Utilities to size-match two datasets by random subsampling.

    The main entry point `subsample(...)` compares dataset sizes from two
    providers (real vs. synthetic) and, if their size ratio exceeds a
    tolerance, returns **wrapped** providers where the larger dataset is
    replaced by a `Subset` of the smaller size. The returned providers are
    `SubsampledProvider` instances that expose the subsampled dataset.

    Notes:
        - Randomness is driven by Python’s `random`. For reproducibility, seed
          it beforehand (e.g., via `SeedingService.set_global_seed(seed)`).
        - The transform passed to `subsample` is used to build datasets for
          measuring lengths and constructing the subsampled dataset. The final
          wrapped provider (a `SubsampledProvider`) already contains the
          subsampled dataset and **ignores** transforms later.
    """

    @staticmethod
    def subsample(
        real_provider: Provider,
        synth_provider: Provider,
        provider_transform: Transform,
        acceptable_ratio: float = 1.1,
    ) -> tuple[Provider, Provider]:
        """Return providers sized to within `acceptable_ratio` by subsampling.

        Compares the lengths of datasets produced by `real_provider` and
        `synth_provider` (using `provider_transform`). If the size ratio
        `max(n_real/n_synth, n_synth/n_real)` is **greater** than
        `acceptable_ratio`, randomly subsamples the **larger** dataset down to
        the size of the smaller one and returns new providers that wrap those
        subsampled datasets. Otherwise, returns the original providers.

        Args:
            real_provider (Provider): Provider for the real dataset.
            synth_provider (Provider): Provider for the synthetic dataset.
            provider_transform (Transform): Transform used to build datasets for
                size computation and for creating the subsampled dataset.
            acceptable_ratio (float, optional): Maximum tolerated size imbalance.
                For example, `1.1` allows up to ±10% difference. Defaults to `1.1`.

        Returns:
            tuple[Provider, Provider]: Potentially updated providers. If
            subsampling occurs, the larger one is replaced by a
            `SubsampledProvider` that wraps a `torch.utils.data.Subset`.

        Raises:
            ValueError: If either dataset is empty.

        Example:
            >>> # If real has 10_000 samples and synth has 5_000:
            >>> # ratio = 10_000/5_000 = 2.0 > 1.1 → subsample real to 5_000
            >>> real_p2, synth_p2 = SubsamplingService.subsample(
            ...     real_provider, synth_provider, provider_transform, acceptable_ratio=1.1
            ... )
        """
        real_data = real_provider.get_data(provider_transform)
        synth_data = synth_provider.get_data(provider_transform)
        real_quantity = len(real_data)
        synth_quantity = len(synth_data)

        ratio = SubsamplingService._calculate_proportion(real_quantity, synth_quantity)
        logger.debug(f"Calculated dataset size ratio: {ratio:.2f}")

        if ratio < acceptable_ratio:
            logger.info(f"No subsampling needed: {ratio:.2f} <= {acceptable_ratio:.2f}")
            return real_provider, synth_provider

        logger.info(f"Subsampling required: {ratio:.2f} > {acceptable_ratio:.2f}")
        if real_quantity > synth_quantity:
            logger.debug("Subsampling the real dataset...")
            indices = random.sample(range(real_quantity), synth_quantity)
            logger.debug(
                f"Selected indices for real dataset: {sorted(indices[:10])}..."
            )
            real_provider = SubsamplingService._get_subsampled_provider(
                real_provider, provider_transform, indices
            )
            logger.debug(f"Subsampled real dataset to {synth_quantity} samples")
        elif synth_quantity > real_quantity:
            logger.debug("Subsampling the synthetic dataset...")
            indices = random.sample(range(synth_quantity), real_quantity)
            logger.debug(
                f"Selected indices for synthetic dataset: {sorted(indices[:10])}..."
            )
            synth_provider = SubsamplingService._get_subsampled_provider(
                synth_provider, provider_transform, indices
            )
            logger.debug(f"Subsampled synthetic dataset to {real_quantity} samples")

        return real_provider, synth_provider

    @staticmethod
    def _calculate_proportion(real_quantity: int, synth_quantity: int) -> float:
        """Return the size imbalance ratio between the two datasets.

        Computes:
            `ratio = max(real_quantity / synth_quantity, synth_quantity / real_quantity)`

        Args:
            real_quantity (int): Number of samples in the real dataset.
            synth_quantity (int): Number of samples in the synthetic dataset.

        Returns:
            float: Size ratio ≥ 1.0 (1.0 means equal sizes).

        Raises:
            ValueError: If either quantity is zero.
        """
        if real_quantity == 0 or synth_quantity == 0:
            logger.error("One of the datasets is empty.")
            raise ValueError("Both datasets must contain at least one sample.")

        ratio = max(real_quantity / synth_quantity, synth_quantity / real_quantity)
        return ratio

    @staticmethod
    def _get_subsampled_provider(
        provider: Provider, transform: Transform, indices: list[int]
    ) -> SubsampledProvider:
        """Wrap the provider's dataset as a `Subset` using the given indices.

        Args:
            provider (Provider): Source provider; used for its `data_path`.
            transform (Transform): Transform used to build the base dataset
                before subsetting.
            indices (list[int]): Sample indices to keep.

        Returns:
            SubsampledProvider: A provider exposing the subsampled dataset.

        Notes:
            - The resulting `SubsampledProvider` **ignores** any future transforms;
              it always returns the preconstructed `Subset`.
        """
        return SubsampledProvider(
            data_path=provider.data_path,
            dataset=Subset(provider.get_data(transform), indices),  # type: ignore
        )
