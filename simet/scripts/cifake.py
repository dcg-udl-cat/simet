"""Utility to download the CIFAKE dataset via KaggleHub.

This module provides a single function, `download_cifake_dataset`, and a
minimal CLI interface:

    python cifake.py <path_to_download>

Notes:
    - Requires `kagglehub` and valid Kaggle credentials/config for the environment.
    - Network I/O and disk writes occur; ensure enough free space.
"""

import sys
from pathlib import Path

from kagglehub import dataset_download


def download_cifake_dataset(path: Path) -> Path:
    """Download the CIFAKE dataset to the given directory.

    Uses KaggleHub to fetch
    `"birdy654/cifake-real-and-ai-generated-synthetic-images"` and stores it
    under `path`. Creates the directory if it does not exist.

    Args:
        path: Destination directory for the dataset.

    Returns:
        Path: The directory where the dataset was downloaded.

    Raises:
        RuntimeError: If the download fails or credentials are missing.
        OSError: For filesystem errors (e.g., permissions, disk full).

    Example:
        >>> download_cifake_dataset(Path("./data/cifake"))  # doctest: +SKIP
        PosixPath('data/cifake')
    """
    path.mkdir(parents=True, exist_ok=True)
    try:
        dataset_download(
            "birdy654/cifake-real-and-ai-generated-synthetic-images",
            path=str(path),
        )
    except Exception as e:  # make failures explicit to callers
        raise RuntimeError(f"Failed to download CIFAKE to {path}: {e}") from e
    return path


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage: python cifake.py <path_to_download>")
        sys.exit(1)
    dest = Path(sys.argv[1])
    download_cifake_dataset(dest)
    print(f"Downloaded CIFAKE to: {dest}")
