import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from PIL import Image

from castinspect.evaluation.metrics import auroc

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Approximate grey of the conveyor background, used to fill rotation corners
# so augmentation does not paint black wedges the real camera never produces.
BACKGROUND_FILL = 200


def build_train_transform(image_size: int) -> Callable[[Image.Image], Any]:
    import torch
    from torchvision.transforms import v2

    # The part is a centred circular casting photographed top-down, so any
    # rotation and both flips are label-preserving; images are grayscale, so
    # no hue or saturation changes.
    transform = v2.Compose(
        [
            v2.Grayscale(num_output_channels=3),
            v2.RandomRotation(degrees=180, fill=BACKGROUND_FILL),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return cast(Callable[[Image.Image], Any], transform)


def build_eval_transform(image_size: int) -> Callable[[Image.Image], Any]:
    import torch
    from torchvision.transforms import v2

    transform = v2.Compose(
        [
            v2.Grayscale(num_output_channels=3),
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return cast(Callable[[Image.Image], Any], transform)


class PathsDataset:
    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable[[Image.Image], Any],
    ) -> None:
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self._paths = list(image_paths)
        self._labels = list(labels)
        self._transform = transform

    @property
    def labels(self) -> list[int]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[Any, float]:
        with Image.open(self._paths[index]) as img:
            image = img.convert("L")
        return self._transform(image), float(self._labels[index])


@dataclass
class TrainSettings:
    max_epochs: int = 15
    patience: int = 3
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 42
    budget_seconds: float | None = None


@dataclass
class TrainResult:
    best_state: dict[str, Any]
    best_metric: float
    best_epoch: int
    epochs_run: int
    epoch_seconds: list[float] = field(default_factory=list)
    history: list[float] = field(default_factory=list)


def _predict_logits(
    network: Any, dataset: PathsDataset, batch_size: int
) -> npt.NDArray[np.float64]:
    import torch
    from torch.utils.data import DataLoader

    loader: DataLoader[Any] = DataLoader(cast(Any, dataset), batch_size=batch_size, shuffle=False)
    network.eval()
    outputs = []
    with torch.no_grad():
        for batch, _ in loader:
            outputs.append(network(batch).squeeze(1))
    return cast(npt.NDArray[np.float64], torch.cat(outputs).numpy().astype(np.float64))


def train_network(
    network: Any,
    train_data: PathsDataset,
    settings: TrainSettings,
    val_data: PathsDataset | None = None,
    epochs_override: int | None = None,
) -> TrainResult:
    """Budgeted CPU training loop with early stopping on validation AUROC.

    With epochs_override set (final model), the loop runs exactly that many
    epochs and keeps the last state instead of selecting on validation.
    """
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(settings.seed)
    torch.use_deterministic_algorithms(True)

    train_labels = np.asarray(train_data.labels, dtype=np.float64)
    n_pos = float(train_labels.sum())
    n_neg = float(len(train_labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training data must contain both classes")
    pos_weight = torch.tensor([n_neg / n_pos])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    max_epochs = epochs_override or settings.max_epochs
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    generator = torch.Generator().manual_seed(settings.seed)
    loader: DataLoader[Any] = DataLoader(
        cast(Any, train_data), batch_size=settings.batch_size, shuffle=True, generator=generator
    )

    best_metric = -np.inf
    best_state: dict[str, Any] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_seconds: list[float] = []
    history: list[float] = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.perf_counter()
        network.train()
        for batch, batch_labels in loader:
            optimizer.zero_grad()
            logits = network(batch).squeeze(1)
            loss = criterion(logits, batch_labels.float())
            loss.backward()
            optimizer.step()
        scheduler.step()
        epoch_seconds.append(time.perf_counter() - epoch_start)

        if val_data is not None and epochs_override is None:
            logits = _predict_logits(network, val_data, settings.batch_size)
            metric = auroc(np.asarray(val_data.labels), logits)
            history.append(metric)
            logger.info("epoch %d: val auroc %.4f (%.0fs)", epoch, metric, epoch_seconds[-1])
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in network.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= settings.patience:
                    logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
                    break
        else:
            best_epoch = epoch
            logger.info("epoch %d done (%.0fs)", epoch, epoch_seconds[-1])

        if settings.budget_seconds is not None:
            elapsed = time.perf_counter() - started
            projected = elapsed + np.mean(epoch_seconds)
            if projected > settings.budget_seconds:
                logger.warning("stopping at epoch %d to respect the time budget", epoch)
                break

    if not best_state:
        best_state = {k: v.clone() for k, v in network.state_dict().items()}
        best_metric = history[-1] if history else float("nan")
        best_epoch = best_epoch or max_epochs
    network.load_state_dict(best_state)
    return TrainResult(
        best_state=best_state,
        best_metric=float(best_metric),
        best_epoch=best_epoch,
        epochs_run=len(epoch_seconds),
        epoch_seconds=epoch_seconds,
        history=history,
    )
