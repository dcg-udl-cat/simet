import torch


class RocAucService:
    """Utilities for feature standardization used in ROC-AUC workflows."""

    @staticmethod
    def standardize_train(
        X: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fit standardization parameters on `X` and return the standardized data.

        Computes per-feature mean and standard deviation over the **batch** and
        returns the standardized tensor along with the fitted parameters.

        Args:
            X (torch.Tensor): Input features of shape ``(n_samples, n_features)``.
                Can be any floating dtype; output matches `X.dtype`.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - ``X_std``: Standardized features, shape ``(n_samples, n_features)``.
                - ``mu``: Per-feature mean, shape ``(1, n_features)``.
                - ``sigma``: Per-feature std (clipped to >= 1e-6), shape ``(1, n_features)``.

        Notes:
            - Uses ``sigma = std.clamp_min(1e-6)`` to avoid division by zero.
            - Statistics are computed along ``dim=0`` with ``keepdim=True`` so they
              broadcast correctly when standardizing.
            - For reproducible pipelines, persist ``mu`` and ``sigma`` for use on
              validation/test sets.
        """
        mu = X.mean(dim=0, keepdim=True)
        sigma = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        return (X - mu) / sigma, mu, sigma

    @staticmethod
    def standardize_with(
        X: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Standardize `X` using provided per-feature mean and std.

        Args:
            X (torch.Tensor): Input features, shape ``(n_samples, n_features)``.
            mu (torch.Tensor): Per-feature mean, shape ``(1, n_features)`` (or broadcastable).
            sigma (torch.Tensor): Per-feature std, shape ``(1, n_features)`` (or broadcastable).
                Should be strictly positive; if computed elsewhere, consider clamping.

        Returns:
            torch.Tensor: Standardized features of the same shape/dtype/device as `X`.

        Notes:
            - `mu` and `sigma` are typically obtained from `standardize_train` on the
              training set and reused for validation/test to avoid data leakage.
            - Relies on PyTorch broadcasting; alternative shapes that broadcast
              (e.g., ``(n_features,)``) are also accepted.
        """
        return (X - mu) / sigma
