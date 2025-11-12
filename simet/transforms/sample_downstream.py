from typing import override

from torchvision import transforms
from torchvision.transforms.transforms import Compose

from simet.transforms import Transform


class SampleDownstreamTransform(Transform):
    """Lightweight preprocessing for downstream tasks (64×64, [-1, 1]).

    Produces a `torchvision.transforms.Compose` that:
      1) resizes the shorter side to 64 px,
      2) center-crops to 64×64,
      3) converts to a tensor,
      4) normalizes channels to approximately ``[-1, 1]`` using mean=0.5/std=0.5.

    Use this for **downstream or toy models** that operate at 64×64, *not* for
    Inception-v3 feature extraction (which expects 299×299 and typically ImageNet
    mean/std).

    Example:
        >>> tfm = SampleDownstreamTransform().get_transform()
        >>> x = tfm(PIL_image)  # tensor shape [3, 64, 64], roughly in [-1, 1]
    """

    @override
    def get_transform(self) -> Compose:
        """Return the 64×64 downstream preprocessing pipeline.

        Returns:
            Compose: Resize→CenterCrop→ToTensor→Normalize(mean=0.5, std=0.5).
        """
        return Compose(
            [
                transforms.Resize(64),        # 64×64 target size for small models
                transforms.CenterCrop(64),    # Ensures centered 64×64 crop
                transforms.ToTensor(),        # PIL → Tensor in [0, 1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # maps to approximately [-1, 1]
            ]
        )
