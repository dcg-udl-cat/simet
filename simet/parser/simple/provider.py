import logging
from pathlib import Path

from simet.providers import LocalProviderWithoutClass, Provider

logger = logging.getLogger(__name__)


class SimpleProviderParser:
    """Minimal provider factory for a single local-unlabeled provider.

    Converts a filesystem path into a :class:`LocalProviderWithoutClass`.
    Useful as a lightweight default when you only need to read images
    without labels (e.g., feature extraction or inference).

    Notes:
        - This parser intentionally supports **only** `LocalProviderWithoutClass`.
          For labeled data or other sources, implement a richer parser/factory
          that selects providers by type.
    """

    @staticmethod
    def parse_provider(provider_path: str) -> Provider:
        """Return a `LocalProviderWithoutClass` for the given path.

        Args:
            provider_path (str): Filesystem path to the image root directory.

        Returns:
            Provider: A `LocalProviderWithoutClass` instance rooted at `provider_path`.

        Example:
            >>> p = SimpleProviderParser.parse_provider("data/unlabeled")
            >>> p.data_path.as_posix().endswith("data/unlabeled")
            True
        """
        data_path = Path(provider_path)
        return LocalProviderWithoutClass(data_path)
