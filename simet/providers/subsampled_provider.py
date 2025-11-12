from pathlib import Path

from torchvision.datasets import VisionDataset

from simet.providers import Provider
from simet.transforms import Transform


class SubsampledProvider(Provider):
    """Provider that wraps an already-built `VisionDataset`.

    This provider is a thin adapter: instead of discovering data on disk,
    it simply returns a preconstructed `VisionDataset` (e.g., a subsampled
    or filtered view) when `get_data(...)` is called. The `transform` argument
    is **ignored on purpose**—assume the wrapped dataset already applies the
    desired preprocessing.

    Args:
        data_path (Path, optional):
            Kept for API parity with `Provider`; not used by this implementation.
        dataset (VisionDataset):
            The dataset instance to expose (e.g., a `Subset`, custom dataset,
            or any `VisionDataset` you prepared externally).

    Attributes:
        dataset (VisionDataset): The wrapped dataset returned by `get_data()`.

    Notes:
        - Because `transform` is ignored, make sure the `dataset` you pass is
          already configured with the correct transforms.
        - `data_path` is retained for uniformity but is not consulted.
        - If `dataset` is omitted, the default `VisionDataset()` placeholder
          will likely raise at runtime; in practice you should always pass a
          concrete dataset.

    Example:
        >>> from torch.utils.data import Subset
        >>> from torchvision.datasets import ImageFolder
        >>> import torchvision.transforms as T
        >>>
        >>> base = ImageFolder("data/train", transform=T.ToTensor())
        >>> small = Subset(base, indices=list(range(1000)))   # subsampled view
        >>> provider = SubsampledProvider(dataset=small)
        >>> ds = provider.get_data(transform=... )  # transform ignored
        >>> len(ds)
        1000
    """

    dataset: VisionDataset

    def __init__(
        self,
        data_path: Path = Path(),
        dataset: VisionDataset = None,  # type: ignore
    ) -> None:
        super().__init__(data_path)
        self.dataset = dataset if dataset else VisionDataset()

    def get_data(self, transform: Transform) -> VisionDataset:
        """Return the wrapped dataset; `transform` is ignored.

        Args:
            transform (Transform): Unused. Present for interface compatibility.

        Returns:
            VisionDataset: The dataset provided at construction time.
        """
        return self.dataset
