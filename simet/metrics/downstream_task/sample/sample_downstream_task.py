import logging
from abc import ABC
from typing import override

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from simet.metrics import DownstreamTask
from simet.models import SimpleCNN

logger = logging.getLogger(__name__)

MY_OPTIM_LR = 1e-3
MY_EPOCHS = 3


class SampleDownstreamTask(DownstreamTask, ABC):
    """Toy downstream task: train a small CNN and report test accuracy.

    Trains a :class:`SimpleCNN` classifier for a few epochs on a provided
    training dataloader and evaluates on a test dataloader, returning the final
    **accuracy** as a scalar. Intended as a lightweight example of how to wire a
    downstream task; not optimized for performance.

    Assumptions:
        - Binary classification (`num_classes=2`) using `CrossEntropyLoss`.
        - Dataloaders yield `(image_tensor, target)` pairs where `target` is
          integer-encoded (0/1).
        - Uses CUDA if available, otherwise CPU.

    Hyperparameters:
        - Optimizer: `Adam(lr=MY_OPTIM_LR)` with `MY_OPTIM_LR = 1e-3`.
        - Epochs: `MY_EPOCHS = 3`.

    Attributes:
        train_set (DataLoader[VisionDataset]): Set during `_compute`.
        test_set (DataLoader[VisionDataset]): Set during `_compute`.
        device (torch.device): `"cuda"` if available else `"cpu"`.
        model (nn.Module): The classifier (initialized in `_compute`).
        criterion (nn.Module): Loss function (`CrossEntropyLoss`).
        optimizer (torch.optim.Optimizer): Adam optimizer on `model` params.
    """

    train_set: DataLoader[VisionDataset]
    test_set: DataLoader[VisionDataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model: nn.Module
    criterion: nn.Module
    optimizer: torch.optim.Optimizer

    @override
    def _compute(
        self, train_set: DataLoader[VisionDataset], test_set: DataLoader[VisionDataset]
    ) -> float:
        """Train for `MY_EPOCHS` and return final test-set accuracy.

        Initializes the model, loss, and optimizer; then runs a simple training
        loop followed by evaluation. Logs per-epoch train loss and test metrics.

        Args:
            train_set: Dataloader used for training.
            test_set: Dataloader used for evaluation.

        Returns:
            float: Final accuracy on the test set in `[0.0, 1.0]`.

        Notes:
            - For reproducibility, seed via your seeding utility before calling.
            - This routine is intentionally minimal (no schedulers/early stopping).
        """
        self.train_set = train_set
        self.test_set = test_set

        self.model = SimpleCNN(num_classes=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=MY_OPTIM_LR)
        logger.debug(f"Declared model, criterion, optimizer on device {self.device}")

        for epoch in range(MY_EPOCHS):
            train_loss = self._train_epoch()
            test_loss, test_acc = self._evaluate()
            logger.debug(
                f"Epoch {epoch + 1}/{MY_EPOCHS} "
                f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                f"test_acc={test_acc:.4f}"
            )
        logger.info(f"Final evaluation on test set after {MY_EPOCHS} epochs")

        _, final_acc = self._evaluate()
        logger.info(f"Final test accuracy: {final_acc:.4f}")
        return final_acc

    def _train_epoch(self) -> float:
        """Run one training epoch over `self.train_set`.

        Performs forward/backward/step for each batch and returns the average
        training loss over the epoch.

        Returns:
            float: Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        seen = 0
        for xb, yb in self.train_set:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(xb)
            loss = self.criterion(logits, yb)
            loss.backward()
            self.optimizer.step()
            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size
        return total_loss / max(1, seen)

    def _evaluate(self) -> tuple[float, float]:
        """Evaluate the model on `self.test_set`.

        Computes average loss and accuracy without gradient tracking.

        Returns:
            tuple[float, float]: `(avg_loss, accuracy)` where `accuracy` is in
            `[0.0, 1.0]`.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in self.test_set:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        avg_loss = total_loss / max(1, total)
        acc = correct / max(1, total)
        return avg_loss, acc
