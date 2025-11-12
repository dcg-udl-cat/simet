from typing import override

from torchvision import transforms
from torchvision.transforms.transforms import Compose

from simet.transforms import Transform


class InceptionTransform(Transform):
    """Preprocessing pipeline for Inception-v3 feature extraction.

    Produces a `torchvision.transforms.Compose` that:
      1) resizes the shorter side to 299 px,
      2) center-crops to 299×299,
      3) converts to a tensor,
      4) normalizes each channel to the range ~[-1, 1] via mean=0.5, std=0.5.

    Notes:
        - The normalization here uses ``mean=[0.5, 0.5, 0.5]`` and
          ``std=[0.5, 0.5, 0.5]`` (mapping 0..1 → −1..1). If you are using
          **torchvision’s** pretrained Inception-v3 weights
          (``Inception_V3_Weights.IMAGENET1K_V1``), the *typical* normalization is
          ImageNet mean/std (``[0.485, 0.456, 0.406]`` / ``[0.229, 0.224, 0.225]``).
          Ensure your transform matches the expectations of the model you use for
          feature extraction to avoid feature drift.
        - The center crop after an exact 299 resize is partly redundant but ensures
          consistent 299×299 framing for non-square inputs.

    Example:
        >>> tfm = InceptionTransform().get_transform()
        >>> img_t = tfm(PIL_image)  # shape [3, 299, 299], roughly in [-1, 1]
    """

    @override
    def get_transform(self) -> Compose:
        """Return the composed preprocessing pipeline for Inception-v3.

        Returns:
            Compose: Resize→CenterCrop→ToTensor→Normalize(mean=0.5, std=0.5).
        """
        return Compose(
            [
                transforms.Resize(299),  # InceptionV3 expects 299x299
                transforms.CenterCrop(299),  # Ensures centered 299×299 crop
                transforms.ToTensor(),  # PIL→Tensor in [0, 1]
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),  # maps to roughly [-1, 1]
            ]
        )
