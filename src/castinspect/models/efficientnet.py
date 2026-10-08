import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from castinspect.data.acquire import sha256_file
from castinspect.models.base import register_model
from castinspect.models.train import (
    PathsDataset,
    TrainSettings,
    build_eval_transform,
    build_train_transform,
    train_network,
)

logger = logging.getLogger(__name__)

TIMM_MODEL = "efficientnet_b0"


class EfficientNetClassifier:
    name = "efficientnet_b0"

    def __init__(
        self,
        pretrained: bool = True,
        weights_sha256: str = "",
        image_size: int = 224,
        batch_size: int = 16,
        max_epochs: int = 15,
        patience: int = 3,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        seed: int = 42,
        budget_seconds: float | None = None,
        epochs_override: int | None = None,
    ) -> None:
        self._pretrained = pretrained
        self._weights_sha256 = weights_sha256
        self._image_size = image_size
        self._settings = TrainSettings(
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            budget_seconds=budget_seconds,
        )
        self._epochs_override = epochs_override
        self._network: Any = None
        self.fit_info: dict[str, Any] = {}

    def _build_network(self) -> Any:
        import timm
        import torch

        torch.manual_seed(self._settings.seed)
        network = timm.create_model(
            TIMM_MODEL, pretrained=self._pretrained, num_classes=1, in_chans=3
        )
        if self._pretrained:
            self._verify_weights_file()
        return network

    def _verify_weights_file(self) -> None:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(f"timm/{TIMM_MODEL}.ra_in1k", "model.safetensors"))
        actual = sha256_file(path)
        if not self._weights_sha256:
            logger.warning("weights checksum not pinned; cached file has sha256 %s", actual)
            return
        if actual != self._weights_sha256:
            raise ValueError(
                f"weights checksum mismatch: expected {self._weights_sha256}, got {actual}"
            )

    def fit(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        val_paths: Sequence[Path] | None = None,
        val_labels: Sequence[int] | None = None,
    ) -> None:
        self._network = self._build_network()
        train_data = PathsDataset(image_paths, labels, build_train_transform(self._image_size))
        val_data = None
        if val_paths is not None and val_labels is not None:
            val_data = PathsDataset(val_paths, val_labels, build_eval_transform(self._image_size))
        result = train_network(
            self._network,
            train_data,
            self._settings,
            val_data=val_data,
            epochs_override=self._epochs_override,
        )
        self.fit_info = {
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_val_auroc": result.best_metric,
            "epoch_seconds": result.epoch_seconds,
            "val_history": result.history,
        }

    def predict_proba(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        import torch
        from torch.utils.data import DataLoader

        if self._network is None:
            raise RuntimeError("model is not fitted")
        dataset = PathsDataset(
            image_paths, [0] * len(image_paths), build_eval_transform(self._image_size)
        )
        loader: DataLoader[Any] = DataLoader(
            cast(Any, dataset), batch_size=self._settings.batch_size, shuffle=False
        )
        self._network.eval()
        outputs = []
        with torch.no_grad():
            for batch, _ in loader:
                outputs.append(torch.sigmoid(self._network(batch).squeeze(1)))
        return cast(npt.NDArray[np.float64], torch.cat(outputs).numpy().astype(np.float64))

    def predict_logits(self, image_paths: Sequence[Path]) -> npt.NDArray[np.float64]:
        import torch
        from torch.utils.data import DataLoader

        if self._network is None:
            raise RuntimeError("model is not fitted")
        dataset = PathsDataset(
            image_paths, [0] * len(image_paths), build_eval_transform(self._image_size)
        )
        loader: DataLoader[Any] = DataLoader(
            cast(Any, dataset), batch_size=self._settings.batch_size, shuffle=False
        )
        self._network.eval()
        outputs = []
        with torch.no_grad():
            for batch, _ in loader:
                outputs.append(self._network(batch).squeeze(1))
        return cast(npt.NDArray[np.float64], torch.cat(outputs).numpy().astype(np.float64))

    def save(self, path: Path) -> None:
        import torch

        if self._network is None:
            raise RuntimeError("model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._network.state_dict(),
                "image_size": self._image_size,
                "pretrained": self._pretrained,
                "weights_sha256": self._weights_sha256,
                "fit_info": self.fit_info,
            },
            path,
        )

    def load(self, path: Path) -> None:
        import timm
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self._image_size = checkpoint["image_size"]
        network = timm.create_model(TIMM_MODEL, pretrained=False, num_classes=1, in_chans=3)
        network.load_state_dict(checkpoint["state_dict"])
        network.eval()
        self._network = network
        self.fit_info = checkpoint.get("fit_info", {})


register_model("efficientnet_b0", EfficientNetClassifier)
