from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import VisionDataset


class NoClassImageDataset(VisionDataset):
    """Unlabeled image dataset loaded from a folder tree.

    Recursively scans the ``root`` directory for image files and returns samples
    as ``(image, target)`` pairs where the **target is always 0** (dummy label).
    Useful for feature extraction, self-supervised preprocessing, or inference
    pipelines that expect a Dataset interface without class folders.

    Supported extensions (case-insensitive): ``.jpg``, ``.jpeg``, ``.png``,
    ``.bmp``, ``.tiff``.

    Args:
        root (Path | str):
            Root directory to scan recursively for images.
        transform (Callable | None):
            Optional transform applied to each PIL image (after RGB conversion).
            Typical usage is a torchvision transform pipeline producing a ``torch.Tensor``.
            If omitted, the raw PIL image is returned.
        target_transform (Callable | None):
            Accepted for API parity with ``VisionDataset`` but **not applied** in this
            implementation because the target is a constant ``0``.

    Attributes:
        image_paths (list[Path]):
            Absolute paths to all discovered image files under ``root``.

    Notes:
        - Directory traversal is **recursive** via ``Path.rglob('*')``.
        - File extension matching is **case-insensitive**.
        - The returned target is the integer ``0``; downstream code should ignore or
          replace it if labels are not needed.
        - If you need a tensor target or to apply ``target_transform``, consider
          wrapping this dataset or subclassing and overriding ``__getitem__``.

    Example:
        >>> ds = NoClassImageDataset("data/images", transform=my_tfms)
        >>> len(ds) > 0
        True
        >>> x, y = ds[0]
        >>> int(y)
        0
    """

    image_paths: list[Path]

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):  # type: ignore
        """Initialize the dataset; see class docstring for parameter details."""
        super().__init__(root, transform=transform, target_transform=target_transform)  # type: ignore
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self) -> list[Path]:
        """Return a list of image file paths discovered under ``root``.

        Scans recursively and filters by a fixed set of image extensions.

        Returns:
            list[Path]: Absolute paths of images found under ``root``.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths: list[Path] = []

        root_path = Path(self.root)
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_paths.append(file_path)

        return image_paths

    def __len__(self) -> int:
        """Return the number of images discovered under ``root``."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Image.Image | torch.Tensor, int]:
        """Load and return the (image, target) pair at index ``idx``.

        The image is opened with PIL and converted to RGB, then ``transform`` is
        applied if provided. The target is the constant integer ``0``.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[PIL.Image.Image | torch.Tensor, int]:
                Transformed image (or raw PIL image if no transform) and the dummy target ``0``.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:  # type: ignore
            image = self.transform(image)  # type: ignore

        return image, 0  # type: ignore  # 0 as dummy target
