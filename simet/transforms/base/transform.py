from abc import ABC, abstractmethod

from torchvision.transforms.transforms import Compose


class Transform(ABC):
    """Abstract interface for building torchvision transform pipelines.

    Implementations should return a `torchvision.transforms.Compose` that
    defines the preprocessing/augmentation steps to apply to each sample
    (e.g., PIL Image → Tensor normalization).

    Subclassing:
        Implement `get_transform()` to construct and return the composed
        transform. Keep any stochastic behavior (e.g., random crops) inside
        the returned pipeline, not in `get_transform()` itself.

    Example:
        >>> import torchvision.transforms as T
        >>> class InceptionTransform(Transform):
        ...     def get_transform(self) -> Compose:
        ...         return T.Compose([
        ...             T.Resize(342),
        ...             T.CenterCrop(299),
        ...             T.ToTensor(),
        ...             T.Normalize(mean=[0.485, 0.456, 0.406],
        ...                          std=[0.229, 0.224, 0.225]),
        ...         ])
    """

    @abstractmethod
    def get_transform(self) -> Compose:
        """Return the composed transform pipeline to apply per sample.

        Returns:
            Compose: A torchvision `Compose` object encapsulating the
            preprocessing/augmentation steps.
        """
        pass
