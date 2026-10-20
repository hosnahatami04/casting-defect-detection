from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

FloatArray = npt.NDArray[np.float64]


class GradCam:
    """Class-activation map from the gradients of the defect logit with respect
    to the last convolutional feature map, captured with hooks."""

    def __init__(self, network: Any, target_layer: str = "conv_head") -> None:
        import torch

        self._network = network
        self._activations: Any = None
        self._gradients: Any = None
        try:
            layer = dict(network.named_modules())[target_layer]
        except KeyError as exc:
            raise ValueError(f"layer {target_layer!r} not found in network") from exc

        def forward_hook(module: Any, args: Any, output: torch.Tensor) -> None:
            self._activations = output.detach()

        def backward_hook(module: Any, grad_input: Any, grad_output: Any) -> None:
            self._gradients = grad_output[0].detach()

        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)

    def heatmap(self, input_tensor: Any) -> FloatArray:
        import torch

        self._network.eval()
        self._network.zero_grad()
        logit = self._network(input_tensor.unsqueeze(0))
        logit.squeeze().backward()
        if self._activations is None or self._gradients is None:
            raise RuntimeError("hooks captured nothing; check the target layer")
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        resized = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        array: FloatArray = resized.numpy().astype(np.float64)
        peak = array.max()
        if peak > 0:
            array = array / peak
        return array


def overlay(image: Image.Image, heatmap: FloatArray, alpha: float = 0.45) -> Image.Image:
    """Heatmap rendered as red intensity over the grayscale image."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    if heatmap.shape != gray.shape:
        heat_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = np.asarray(heat_img.resize(image.size, Image.Resampling.BILINEAR)) / 255.0
    red = np.clip(gray * (1 - alpha) + 255 * alpha * heatmap, 0, 255)
    other = np.clip(gray * (1 - alpha * heatmap), 0, 255)
    stacked = np.stack([red, other, other], axis=2).astype(np.uint8)
    return Image.fromarray(stacked, mode="RGB")
