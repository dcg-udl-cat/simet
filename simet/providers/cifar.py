from typing import override

from torchvision.datasets import CIFAR10, VisionDataset

from simet.providers import Provider
from simet.transforms import Transform


class CIFARProvider(Provider):
    """Provider that exposes the **CIFAR-10 training split** via `VisionDataset`.

    Wraps `torchvision.datasets.CIFAR10` with `train=True`, `download=True`, and
    applies the project `Transform`. Data will be downloaded (if missing) into
    `self.data_path`.

    Notes:
        - This provider is fixed to the **training** split (`train=True`). If you
          need the test/validation split, create a separate provider or make the
          split configurable.
        - The given `Transform` should handle any normalization/augmentation
          appropriate for CIFAR-10 (e.g., mean/std, random crops/flips).

    Example:
        >>> provider = CIFARProvider(data_path=Path("./data/cifar10"))
        >>> ds = provider.get_data(transform=SomeTransform())
        >>> len(ds)  # 50,000 images in the training split
        50000
    """

    @override
    def get_data(self, transform: Transform) -> VisionDataset:
        """Return the CIFAR-10 **training** dataset with the provided transform.

        Args:
            transform: Project transform wrapper; `transform.get_transform()` is
                passed to the underlying `CIFAR10` dataset.

        Returns:
            VisionDataset: A CIFAR-10 training dataset ready for a `DataLoader`.
        """
        return CIFAR10(
            root=self.data_path,
            train=True,
            download=True,
            transform=transform.get_transform(),
        )
