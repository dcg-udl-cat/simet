import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SeedingService:
    """Utilities for end-to-end reproducibility across Python, NumPy, and PyTorch.

    This service sets seeds for Python's `random`, NumPy, and PyTorch (CPU & CUDA),
    configures CuDNN for deterministic behavior, and applies environment variables
    commonly recommended for reproducible runs.
    """

    @staticmethod
    def set_global_seed(seed: int = 42) -> None:
        """Set global RNG seeds and enable deterministic modes where possible.

        Steps performed:
            1) Seed Python's `random`, NumPy, and PyTorch (CPU + CUDA, all devices).
            2) Configure CuDNN for determinism (`deterministic=True`, `benchmark=False`).
            3) Set environment variables:
               - `PYTHONHASHSEED` → ensure stable hashing in Python.
               - `CUBLAS_WORKSPACE_CONFIG=":4096:8"` → enforce determinism in cuBLAS.
            4) Enable PyTorch deterministic algorithms:
               - Prefer `torch.use_deterministic_algorithms(True)`.
               - Fallbacks and warn-only mode are attempted for older versions or
                 when unsupported ops are encountered.

        Args:
            seed (int): The seed value to apply across libraries. Defaults to 42.

        Notes:
            - **Performance trade-off**: Deterministic CuDNN and algorithms can
              slow down training/inference and increase memory usage.
            - **Coverage limits**: Some PyTorch ops are inherently nondeterministic
              or do not have deterministic implementations. In such cases,
              PyTorch may raise a `RuntimeError`. This method catches it and
              attempts `warn_only=True` when available; otherwise it logs a warning.
            - **Multi-GPU**: Both `torch.cuda.manual_seed` and
              `torch.cuda.manual_seed_all` are set to cover multiple devices.
            - **Version compatibility**: On very old PyTorch versions,
              `torch.use_deterministic_algorithms` may be missing; a fallback to
              `torch.set_deterministic(True)` is attempted (deprecated in newer versions).

        Example:
            >>> SeedingService.set_global_seed(123)
            >>> # All subsequent RNG calls should now be repeatable (within limits).
        """
        logger.info(f"Setting global seed to {seed}")

        # Python's built-in random
        random.seed(seed)
        logger.debug(f"Python random seed set to {seed}")

        # NumPy
        np.random.seed(seed)
        logger.debug(f"NumPy random seed set to {seed}")

        # PyTorch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
        logger.debug(f"PyTorch random seed set to {seed}")

        # PyTorch deterministic operations (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("PyTorch set to use deterministic algorithms where possible")

        # Set environment variables for additional reproducibility
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        logger.debug("Environment variables for reproducibility set")

        # Enable PyTorch deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True)
            logger.debug("PyTorch set to use deterministic algorithms")
        except AttributeError:
            # Fallback for older PyTorch versions
            logger.warning(
                "torch.use_deterministic_algorithms not available, falling back to torch.set_deterministic"
            )
            torch.set_deterministic(True)  # type: ignore
        except RuntimeError as e:
            # Some operations don't support deterministic mode
            logger.warning(f"Could not enable full deterministic mode: {e}")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                logger.debug(
                    "PyTorch set to use deterministic algorithms with warnings only"
                )
            except (AttributeError, TypeError):
                logger.warning(
                    "Could not set warn_only mode for deterministic algorithms"
                )
                pass

        logger.info(f"Global seed set to {seed} for reproducible results")
