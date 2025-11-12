import logging
from typing import override

from torchvision.datasets import VisionDataset

from simet.datasets import NoClassImageDataset
from simet.providers import Provider
from simet.transforms import Transform

logger = logging.getLogger(__name__)


class LocalProviderWithoutClass(Provider):
    """Provider that exposes **unlabeled** images from a local folder tree.

    Wraps :class:`simet.datasets.NoClassImageDataset` to read all images found
    recursively under ``self.data_path`` and apply the given project
    :class:`Transform`. Targets are a **dummy 0** for every sample (no classes).

    Notes:
        - Use when you only need images (e.g., feature extraction/inference) and
          labels are irrelevant.
        - File discovery is recursive; supported extensions are defined by
          :class:`NoClassImageDataset`.

    Example:
        >>> provider = LocalProviderWithoutClass(data_path=Path("./data/unlabeled"))
        >>> ds = provider.get_data(transform=SomeTransform())
        >>> x, y = ds[0]
        >>> int(y)
        0
    """

    @override
    def get_data(self, transform: Transform) -> VisionDataset:
        """Return a `NoClassImageDataset` for `self.data_path` with `transform`.

        Validates that `self.data_path` exists before constructing the dataset.

        Args:
            transform: Project transform wrapper; `transform.get_transform()` is
                passed to the underlying dataset.

        Returns:
            VisionDataset: A dataset yielding `(image, 0)` pairs (dummy target).

        Raises:
            FileNotFoundError: If `self.data_path` does not exist.
            RuntimeError: If no images are found under `self.data_path`
                (raised by `NoClassImageDataset`).
        """
        if not self.data_path.exists():
            logger.error(f"The specified path {self.data_path} does not exist.")
            raise FileNotFoundError(
                f"The specified path {self.data_path} does not exist."
            )
        dataset = NoClassImageDataset(
            root=self.data_path, transform=transform.get_transform()
        )
        logger.info(f"Created dataset from {self.data_path}")
        return dataset
