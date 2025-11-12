import logging
from typing import override

import numpy as np
import torch

from simet.dataset_loaders import DatasetLoader
from simet.metrics import Metric

logger = logging.getLogger(__name__)


class FID(Metric[float]):
    """Frechet Inception Distance computed from pre-extracted features.

    Expects 2D feature arrays for **real** and **synthetic** datasets with the
    same dimensionality (``D``). Computes per-set mean and unbiased covariance,
    then evaluates:

        FID = ||μ_r − μ_s||² + Tr(Σ_r + Σ_s − 2·(Σ_r^{1/2} Σ_s Σ_r^{1/2})^{1/2})

    Numerical behavior:
        - Uses double precision (`float64`) for statistics by default.
        - Runs linear algebra on CUDA if available; otherwise on CPU.
        - Adds a small diagonal (`base_eps`) to covariance matrices.
        - Tries a pure-PyTorch eigen/sqrt path with escalating eps; falls back
          to SciPy’s `sqrtm` if needed.

    Args:
        base_eps (float, default=1e-6): Diagonal regularization added to
            covariance matrices. Also used as a minimum eigenvalue floor.

    Attributes:
        _LINALG_DEVICE (torch.device): Device for linear algebra ops.
        _STAT_DTYPE (torch.dtype): Dtype for statistics (default: float64).
        _BASE_EPS (float): Stabilizing epsilon for covariances/eigenvalues.

    Raises:
        ValueError: If feature arrays are not 2D, have mismatched dims, or have
            fewer than two samples (unbiased covariance requires n≥2).
    """

    def __init__(self, base_eps: float = 1e-6) -> None:
        logger.info("Initializing FID metric")
        super().__init__()
        self._LINALG_DEVICE = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._STAT_DTYPE = torch.float64
        self._BASE_EPS = float(base_eps)

    @property
    @override
    def name(self) -> str:
        """Human-readable metric name."""
        return "FID"

    @override
    def compute(self, loader: DatasetLoader) -> float:
        """Compute FID from `loader.real_features` and `loader.synth_features`.

        Steps:
            1) Move features to `_LINALG_DEVICE` in `_STAT_DTYPE`.
            2) Validate shapes: both 2D and same feature dimension.
            3) Compute `(μ, Σ)` for real and synthetic sets.
            4) Compute the sandwich covariance square root and final FID.

        Args:
            loader (DatasetLoader): Must expose `real_features` and
                `synth_features` as 2D arrays with the same second dimension.

        Returns:
            float: Non-negative FID score (lower is better).

        Raises:
            ValueError: If shapes are invalid or sample counts are < 2.
        """
        with torch.no_grad():
            real = torch.as_tensor(
                loader.real_features, device=self._LINALG_DEVICE, dtype=self._STAT_DTYPE
            )
            logger.debug("Loaded real features into tensor")
            synth = torch.as_tensor(
                loader.synth_features,
                device=self._LINALG_DEVICE,
                dtype=self._STAT_DTYPE,
            )
            logger.debug("Loaded synthetic features into tensor")

            logger.debug("Checking feature shapes and types")
            if real.ndim != 2 or synth.ndim != 2:
                logger.error("Features must be 2D [num_samples, feat_dim].")
                raise ValueError("Features must be 2D [num_samples, feat_dim].")
            if real.size(1) != synth.size(1):
                logger.error(
                    "Real and synthetic features must have the same dimensionality."
                )
                raise ValueError(
                    "Real and synthetic features must have the same dimensionality."
                )
            if real.size(0) < 2 or synth.size(0) < 2:
                logger.error(
                    "Need at least two samples per set to compute unbiased covariance."
                )
                raise ValueError(
                    "Need at least two samples per set to compute unbiased covariance."
                )

            real_mu, real_sigma = self._compute_statistics(real)
            synth_mu, synth_sigma = self._compute_statistics(synth)
            logger.info("Computed statistics for real and synthetic features")

            fid = self._compute_fid(real_mu, real_sigma, synth_mu, synth_sigma)
            logger.info(f"Computed FID: {fid}")

            return fid

    def _compute_statistics(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and unbiased covariance (with symmetric + eps correction).

        Args:
            features: Tensor of shape `(N, D)` in `_STAT_DTYPE`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: `(mu, sigma)` where
                `mu` has shape `(D,)` and `sigma` is `(D, D)`.
        """
        mu = features.mean(dim=0)
        x = features - mu
        n = x.shape[0]
        sigma = (x.T @ x) / (n - 1)
        sigma = 0.5 * (sigma + sigma.T)
        d = sigma.size(0)
        if self._BASE_EPS > 0.0:
            sigma = sigma + self._BASE_EPS * torch.eye(
                d, device=sigma.device, dtype=sigma.dtype
            )
        return mu, sigma

    def _compute_fid(
        self,
        real_mu: torch.Tensor,
        real_sigma: torch.Tensor,
        synth_mu: torch.Tensor,
        synth_sigma: torch.Tensor,
    ) -> float:
        """Compute FID from means and covariances."""
        diff = real_mu - synth_mu
        covmean = self._cov_sqrt_sandwich_auto(real_sigma, synth_sigma)
        trace_term = torch.trace(real_sigma + synth_sigma - 2.0 * covmean)
        fid = diff.dot(diff) + trace_term
        fid = torch.clamp(fid, min=0.0)
        return float(fid.item())

    def _cov_sqrt_sandwich_auto(
        self, sr: torch.Tensor, ss: torch.Tensor
    ) -> torch.Tensor:
        """Compute `(Σ_r^{1/2} Σ_s Σ_r^{1/2})^{1/2}` with robust fallbacks.

        Tries a PyTorch eigen/sqrt path using `_cov_sqrt_sandwich_torch` and
        escalates `eps` up to 3 times; if any result contains non-finite values,
        falls back to a SciPy-based implementation.

        Args:
            sr: Real covariance `(D, D)`.
            ss: Synthetic covariance `(D, D)`.

        Returns:
            torch.Tensor: Symmetric PSD matrix `(D, D)`.
        """
        eps = self._BASE_EPS
        for _ in range(3):
            cov = self._cov_sqrt_sandwich_torch(sr, ss, eps)
            if torch.isfinite(cov).all():
                return cov
            eps *= 10.0
        return self._cov_sqrt_sandwich_scipy(sr, ss, eps)

    def _cov_sqrt_sandwich_torch(
        self, sr: torch.Tensor, ss: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """Torch-only path for the sandwich covariance square root."""
        d = sr.size(0)
        sr_reg = sr + eps * torch.eye(d, device=sr.device, dtype=sr.dtype)
        ss_reg = ss + eps * torch.eye(d, device=ss.device, dtype=ss.dtype)

        # A = Σr^{1/2}
        A = self._spd_sqrt_eig(sr_reg)
        # B = A Σs A
        B = A @ ss_reg @ A
        B = 0.5 * (B + B.T)
        # covmean = (A Σs A)^{1/2}
        covmean = self._spd_sqrt_eig(B)
        return 0.5 * (covmean + covmean.T)

    def _cov_sqrt_sandwich_scipy(
        self, sr: torch.Tensor, ss: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """SciPy fallback for the sandwich covariance square root.

        Uses `scipy.linalg.sqrtm` on CPU with `numpy` arrays and converts back
        to a torch tensor on the original device/dtype.

        Args:
            sr: Real covariance `(D, D)`.
            ss: Synthetic covariance `(D, D)`.
            eps: Diagonal regularization applied before square roots.

        Returns:
            torch.Tensor: Symmetric PSD matrix `(D, D)`.
        """
        import scipy.linalg as la

        a = sr.detach().cpu().numpy()
        b = ss.detach().cpu().numpy()
        d = a.shape[0]

        a_reg = a + eps * np.eye(d, dtype=a.dtype)
        b_reg = b + eps * np.eye(d, dtype=b.dtype)

        # A = sqrtm(Σr)
        A = np.real(la.sqrtm(a_reg))
        # B = A Σs A
        B = A @ b_reg @ A
        B = 0.5 * (B + B.T)
        # covmean = sqrtm(B)  (NO resandwiching with A)
        covmean_np = np.real(la.sqrtm(B))
        covmean_np = 0.5 * (covmean_np + covmean_np.T)

        return torch.from_numpy(covmean_np).to(sr.device, dtype=sr.dtype)

    def _spd_sqrt_eig(self, M: torch.Tensor) -> torch.Tensor:
        """Matrix square root of a symmetric positive (semi)definite matrix.

        Computes `V diag(sqrt(clamp(λ))) Vᵀ` from the eigen-decomposition of the
        symmetrized input. Eigenvalues are floored by `max(base_eps, eps * scale)`
        to avoid negative/zero values due to numerical noise.

        Args:
            M: Symmetric (or nearly symmetric) PSD matrix `(D, D)`.

        Returns:
            torch.Tensor: Symmetric square-root matrix `(D, D)`.
        """
        S = 0.5 * (M + M.T)
        w, V = torch.linalg.eigh(S)
        tiny = torch.finfo(M.dtype).eps
        scale = (torch.trace(S) / S.shape[0]).abs().item() if S.shape[0] > 0 else 1.0
        floor = max(self._BASE_EPS, tiny * max(1.0, scale))
        w = torch.clamp(w, min=floor)
        sqrt_w = torch.sqrt(w)
        return (V * sqrt_w.unsqueeze(0)) @ V.T
