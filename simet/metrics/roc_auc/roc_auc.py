import logging
import math
from typing import override

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from simet.dataset_loaders import DatasetLoader
from simet.metrics import Metric
from simet.metrics.roc_auc.mlp import RocMLP
from simet.services import RocAucService

logger = logging.getLogger(__name__)


class RocAuc(Metric[float]):
    """Binary ROC-AUC metric using a small MLP with K-fold cross-validation.

    Treats **real** features as label `1` and **synthetic** features as label `0`,
    trains a lightweight MLP (logit output) on standardized features with
    stratified K-fold CV, and reports the **mean ROC-AUC** across folds.

    Pipeline (per fold):
        1) Concatenate real/synth features → `(N, D)`, build labels (1/0).
        2) Split with `StratifiedKFold(n_splits=kfolds)`.
        3) Standardize **using train stats only** (via `RocAucService`).
        4) Train `RocMLP` with BCE-with-logits, AdamW, optional AMP, early stopping
           by patience on validation ROC-AUC.
        5) Record the **best** validation ROC-AUC for the fold.
        6) Return the mean (and log the std).

    Args:
        kfolds (int, default=5): Number of stratified folds.
        batch_size (int, default=1024): Minibatch size for training/validation.
        epochs (int, default=100): Max epochs per fold (early-stopped by `patience`).
        patience (int, default=10): Early-stop patience in epochs on val ROC-AUC.
        lr (float, default=1e-3): Learning rate for AdamW.
        weight_decay (float, default=1e-4): L2 weight decay for AdamW.
        hidden1 (int, default=256): First hidden layer width for `RocMLP`.
        hidden2 (int, default=128): Second hidden layer width for `RocMLP`.
        dropout (float, default=0.2): Dropout probability after each hidden layer.
        use_amp (bool | None, default=None):
            Mixed precision flag. `None` → auto-enable on CUDA, else use the given value.
        verbose (bool, default=False): Reserved for future verbose logging.

    Attributes:
        kfolds: Mirror constructor argument.
        batch_size: Mirror constructor argument.
        epochs: Mirror constructor argument.
        patience: Mirror constructor argument.
        lr: Mirror constructor argument.
        weight_decay: Mirror constructor argument.
        hidden1: Mirror constructor argument.
        hidden2: Mirror constructor argument.
        dropout: Mirror constructor argument.
        use_amp: Mirror constructor argument.
        verbose: Mirror constructor argument.
        _last_mean (float): Mean ROC-AUC from the most recent `compute` call.
        _last_std (float): Std of ROC-AUC across folds from the most recent call.

    Requirements:
        - `loader.real_features` and `loader.synth_features` must be 2D arrays
          of shape `(N_real, D)` and `(N_synth, D)` with the **same** feature dimension `D`.

    Example:
        >>> metric = RocAuc(kfolds=5, epochs=50, patience=5)
        >>> score = metric.compute(loader)
        >>> 0.0 <= score <= 1.0
        True
    """

    def __init__(
        self,
        kfolds: int = 5,
        batch_size: int = 1024,
        epochs: int = 100,
        patience: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.2,
        use_amp: bool | None = None,  # None -> auto(use if cuda)
        verbose: bool = False,
    ) -> None:
        logger.info("Initializing ROC AUC metric")
        super().__init__()
        self.kfolds = kfolds
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dropout = dropout
        self.use_amp = use_amp
        self.verbose = verbose

    @property
    @override
    def name(self) -> str:
        """Human-readable metric name."""
        return "ROC AUC"

    @override
    def compute(self, loader: DatasetLoader) -> float:
        """Compute mean ROC-AUC via stratified K-fold training/validation.

        Steps:
            - Convert `loader.real_features` (label=1) and `loader.synth_features` (label=0)
              to float32 tensors; shuffle once for fold assignment.
            - For each fold: standardize with train stats, train `RocMLP` with BCE-with-logits,
              validate with ROC-AUC, and track the best AUC with patience-based early stopping.
            - Return the mean ROC-AUC across folds; log the standard deviation.

        Args:
            loader (DatasetLoader): Must expose `real_features` and `synth_features`
                as 2D arrays with identical feature dimension.

        Returns:
            float: Mean ROC-AUC across folds.

        Raises:
            ValueError: If features are not 2D or do not share the same feature dimension.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device: {device} for ROC AUC computation")
        use_amp = self.use_amp if self.use_amp is not None else device.type == "cuda"

        # --- tensors (float32 is enough here)
        real = torch.as_tensor(loader.real_features, dtype=torch.float32)
        synth = torch.as_tensor(loader.synth_features, dtype=torch.float32)
        logger.debug("Loaded features into tensors")
        if not real.ndim == 2 and synth.ndim == 2 and real.shape[1] == synth.shape[1]:
            logger.error(
                "Features must be 2D [num_samples, feat_dim] and match in feat_dim"
            )
            raise ValueError(
                "Features must be 2D [num_samples, feat_dim] and match in feat_dim"
            )

        X = torch.cat([real, synth], dim=0)
        y = torch.cat(
            [
                torch.ones(len(real), dtype=torch.float32),
                torch.zeros(len(synth), dtype=torch.float32),
            ],
            dim=0,
        )

        # Shuffle once in CPU space for fold assignment
        logger.debug("Shuffling data for K-Fold assignment")
        N = X.shape[0]
        idx = torch.randperm(N)
        X, y = X[idx], y[idx]

        # K-fold (stratified)
        logger.debug("Starting K-Fold cross-validation")
        y_np = y.numpy()
        skf = StratifiedKFold(n_splits=self.kfolds, shuffle=False)
        aucs = []

        for tr_idx, va_idx in skf.split(np.zeros(N), y_np):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            # Standardize using TRAIN stats only
            X_tr, mu, sigma = RocAucService.standardize_train(X_tr)
            X_va = RocAucService.standardize_with(X_va, mu, sigma)

            ds_tr = TensorDataset(X_tr, y_tr)
            ds_va = TensorDataset(X_va, y_va)
            dl_tr = DataLoader(
                ds_tr, batch_size=self.batch_size, shuffle=True, pin_memory=True
            )
            dl_va = DataLoader(
                ds_va, batch_size=self.batch_size, shuffle=False, pin_memory=True
            )

            model = RocMLP(
                in_dim=X.shape[1],
                h1=self.hidden1,
                h2=self.hidden2,
                p=self.dropout,
            ).to(device)
            logger.debug(f"Initialized MLP model: {model}")

            optim = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore
            logger.debug("Initialized optimizer, loss function, and gradient scaler")

            best_auc = -math.inf
            best_state = None
            bad = 0

            for _ in range(self.epochs):
                model.train()
                running = 0.0
                for xb, yb in dl_tr:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)

                    optim.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    running += loss.item()

                # validate
                model.eval()
                with torch.no_grad():
                    all_logits = []
                    all_targets = []
                    for xb, yb in dl_va:
                        xb = xb.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                            logits = model(xb)
                        all_logits.append(logits.detach().cpu())
                        all_targets.append(yb)

                logits_cpu = torch.cat(all_logits).numpy()
                targets_cpu = torch.cat(all_targets).numpy()
                auc = roc_auc_score(
                    targets_cpu, logits_cpu
                )  # logits OK; roc_auc is monotonic wrt sigmoid

                if auc > best_auc + 1e-4:
                    best_auc = auc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

            # restore best (not strictly necessary since we already recorded best_auc)
            logger.debug(f"Best AUC for this fold: {best_auc}")
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            aucs.append(float(best_auc))

        # Final score: mean AUC across folds (optionally you could return mean±std)
        logger.debug(f"ROC AUC scores across folds: {aucs}")
        self._last_mean = float(np.mean(aucs))
        self._last_std = float(np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0)

        logger.info(f"Mean ROC AUC: {self._last_mean} ± {self._last_std}")
        return self._last_mean
