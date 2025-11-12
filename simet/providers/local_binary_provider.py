import logging
from typing import override

from torchvision.datasets import VisionDataset

from simet.datasets import BinaryFolderDataset
from simet.providers import Provider
from simet.transforms import Transform

logger = logging.getLogger(__name__)


class LocalBinaryProvider(Provider):
    """Provider that builds a binary image dataset from a local folder tree.

    Wraps :class:`simet.datasets.BinaryFolderDataset` to read images from
    ``self.data_path`` with the expected structure (case-insensitive class dirs):
    **GOOD/** and **BAD/**. Applies the given project :class:`Transform`.

    Notes:
        - Labels are assigned as: GOOD → 1, BAD → 0.
        - File discovery is recursive under the class subfolders.
        - If the path does not exist or contains no images, an exception is raised.

    Example:
        >>> provider = LocalBinaryProvider(data_path=Path("./data/my_run"))
        >>> ds = provider.get_data(transform=SomeTransform())
        >>> len(ds) > 0
        True
    """

    @override
    def get_data(self, transform: Transform) -> VisionDataset:
        """Return a `BinaryFolderDataset` for `self.data_path` with `transform`.

        Validates that `self.data_path` exists before constructing the dataset.

        Args:
            transform: Project transform wrapper; `transform.get_transform()` is
                passed to the underlying dataset.

        Returns:
            VisionDataset: A dataset yielding `(image, label)` pairs.

        Raises:
            FileNotFoundError: If `self.data_path` does not exist.
            RuntimeError: If no images are found under GOOD/ and BAD/ (raised by
                `BinaryFolderDataset`).
        """
        if not self.data_path.exists():
            logger.error(f"The specified path {self.data_path} does not exist.")
            raise FileNotFoundError(
                f"The specified path {self.data_path} does not exist."
            )
        dataset = BinaryFolderDataset(
            root=self.data_path, transform=transform.get_transform()
        )
        logger.info(f"Created dataset from {self.data_path}")
        return dataset
