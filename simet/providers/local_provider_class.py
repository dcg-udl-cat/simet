from typing import override

from torchvision.datasets import ImageFolder, VisionDataset

from simet.providers import Provider
from simet.transforms import Transform


class LocalProviderWithClass(Provider):
    """Provider that exposes a labeled dataset from a local folder hierarchy.

    Wraps `torchvision.datasets.ImageFolder` to read images from `self.data_path`
    where each **subdirectory name is a class label** (e.g., `cats/`, `dogs/`).
    Applies the given project :class:`Transform`.

    Expected layout:
        data_path/
          class_a/
            img001.jpg
            ...
          class_b/
            img101.jpg
            ...

    Notes:
        - Class-to-index mapping is derived from subfolder names
          (`dataset.class_to_idx`).
        - Discovery is recursive under each class subfolder.
        - If the path does not exist or no valid images are found, `ImageFolder`
          will raise.

    Example:
        >>> provider = LocalProviderWithClass(data_path=Path("./data/pets"))
        >>> ds = provider.get_data(transform=SomeTransform())
        >>> len(ds) > 0
        True
        >>> ds.classes  # e.g., `['cats', 'dogs']`
        ['cats', 'dogs']
    """

    @override
    def get_data(self, transform: Transform) -> VisionDataset:
        """Return an `ImageFolder` dataset rooted at `self.data_path`.

        Validates that `self.data_path` exists before constructing the dataset.

        Args:
            transform: Project transform wrapper; `transform.get_transform()` is
                passed to `ImageFolder`.

        Returns:
            VisionDataset: A dataset yielding `(image, class_index)` pairs.

        Raises:
            FileNotFoundError: If `self.data_path` does not exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"The specified path {self.data_path} does not exist."
            )
        return ImageFolder(
            root=str(self.data_path), transform=transform.get_transform()
        )
