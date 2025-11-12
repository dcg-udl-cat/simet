from abc import ABC, abstractmethod
from pathlib import Path

from torchvision.datasets import VisionDataset

from simet.transforms import Transform


class Provider(ABC):
    """Abstract data provider that builds a `VisionDataset` from a path + transform.

    A `Provider` encapsulates how data is discovered and exposed as a
    `torchvision.datasets.VisionDataset`. It takes care of reading from
    `data_path` (e.g., folder layout, metadata) and applying a project-defined
    `Transform` to produce a dataset consumable by PyTorch `DataLoader`s.

    Args:
        data_path (Path):
            Root path where the underlying data resides (e.g., a folder tree).

    Attributes:
        data_path (Path): The root path provided at construction.

    Subclassing:
        Implement `get_data(transform)` to return a `VisionDataset` instance
        that will yield `(image, target)` pairs using the given `Transform`.
        The returned dataset should be ready to plug into a `DataLoader`.

    Example:
        >>> from torchvision.datasets import ImageFolder
        >>> import torchvision.transforms as T
        >>>
        >>> class FolderProvider(Provider):
        ...     def get_data(self, transform: Transform) -> VisionDataset:
        ...         tfm = transform.get_transform()
        ...         return ImageFolder(self.data_path, transform=tfm)
        ...
        >>> provider = FolderProvider(Path("data/train"))
        >>> dataset = provider.get_data(transform=SomeTransform())
        >>> len(dataset) >= 0
        True
    """

    data_path: Path

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path

    @abstractmethod
    def get_data(self, transform: Transform) -> VisionDataset:
        """Return a `VisionDataset` configured with the given transform.

        Implementations should construct and return a dataset that reads from
        `self.data_path` and applies `transform.get_transform()` to each sample.

        Args:
            transform (Transform): Project transform wrapper used to build the
                underlying `torchvision.transforms.Compose`.

        Returns:
            VisionDataset: Dataset yielding `(sample, target)` pairs suitable for
            consumption by a `torch.utils.data.DataLoader`.
        """
        pass
