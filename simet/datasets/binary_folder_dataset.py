import logging
import os
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torchvision.datasets import VisionDataset

logger = logging.getLogger(__name__) 

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}

class BinaryFolderDataset(VisionDataset):
    """Binary image dataset sourced from a folder structure.

    Expects the root directory to contain two subdirectories (case-insensitive):
    **GOOD/** and **BAD/**. All image files under these subtrees are discovered
    recursively and labeled as:
      - GOOD → label **1**
      - BAD  → label **0**

    Supported extensions include: .jpg, .jpeg, .png, .bmp, .gif, .webp, .tif, .tiff.

    Args:
        root (Path | str):
            Root directory of the dataset. Must contain subfolders for the two classes
            (e.g., ``GOOD/`` and ``BAD/``; case-insensitive).
        transform (Callable | None):
            Optional transform applied to each PIL image before it is returned.
            Typical usage is a torchvision transform pipeline that produces a
            ``torch.Tensor``. If omitted, the raw PIL image (converted to RGB) is returned.
        target_transform (Callable | None):
            Optional transform applied to the integer class label (0/1) **after**
            it is created. (Note: if not used in ``__getitem__``, this parameter is
            accepted for API parity with ``VisionDataset``.)

    Attributes:
        root (Path | str):
            The dataset root provided at construction.
        samples (list[tuple[Path, int]]):
            List of discovered samples as (filepath, label) tuples, where label is
            1 for GOOD and 0 for BAD. The list is shuffled at initialization.

    Raises:
        FileNotFoundError:
            If ``root`` does not exist.
        RuntimeError:
            If no images are found under the expected class subdirectories.

    Notes:
        - Class folder names are matched case-insensitively via lower-casing.
        - Directory walking is recursive; all nested images under GOOD/BAD are included.
        - The sample list is shuffled once at initialization (not per epoch).
        - If ``transform`` is ``None``, ``__getitem__`` returns a PIL image (RGB) instead
          of a tensor; be consistent with your downstream code.

    Example:
        >>> ds = BinaryFolderDataset("data/my_dataset", transform=my_tfms)
        >>> len(ds)
        1280
        >>> x, y = ds[0]
        >>> x.shape  # doctest: +SKIP
        torch.Size([3, H, W])
        >>> int(y)
        1
    """

    def __init__(
        self,
        root: Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:  # type: ignore
        """Initialize the dataset; see class docstring for parameter details."""
        super().__init__(root, transform=transform, target_transform=target_transform)  # type: ignore
        self.root = Path(root)
        if not self.root.exists():
            logger.error(f"Dataset root {self.root} does not exist.")
            raise FileNotFoundError(f"Dataset root {self.root} does not exist.")

        self.samples: list[tuple[Path, int]] = []
        for dirpath, dirnames, _ in os.walk(self.root):
            for dirname in dirnames:
                label_name = dirname.lower()
                if label_name not in ("good", "bad"):
                    continue
                label = 1 if label_name == "good" else 0
                current = Path(dirpath) / dirname
                for candidate in current.rglob("*"):
                    if (
                        candidate.is_file()
                        and candidate.suffix.lower() in IMG_EXTENSIONS
                    ):
                        self.samples.append((candidate, label))

        if not self.samples:
            logger.error(
                f"No images found under {self.root}. Expected GOOD/ and BAD/ subdirectories."
            )
            raise RuntimeError(
                f"No images found under {self.root}. Expected GOOD/ and BAD/ subdirectories."
            )

        random.shuffle(self.samples)

    def __len__(self) -> int:  # type: ignore[override]
        """Return the number of samples discovered under GOOD/BAD."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Load and return a (image, label) pair at the given index.

        The image is opened with PIL and converted to RGB. If ``transform`` is set,
        it is applied to the image; otherwise the raw PIL image is returned.
        The label is returned as a ``torch.long`` tensor with values {0, 1}.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor | PIL.Image.Image, torch.Tensor]:
                The transformed image (or PIL image if no transform) and the label tensor.

        """
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        tensor = self.transform(image) if self.transform else image
        target = torch.tensor(label, dtype=torch.long)
        return tensor, target  # type: ignore
