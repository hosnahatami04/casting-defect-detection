"""Preprocessing shared by every inference path.

The API, the ONNX runtime and the ONNX parity test all call preprocess() so
there is exactly one definition of how a raw image becomes a model input. It is
pure numpy with no torch dependency, so the service image does not need torch.
The steps mirror the eval transform used in training (grayscale, resize to
224, scale to [0, 1], ImageNet normalisation, grayscale replicated to three
channels); the parity test pins them against the torchvision transform.
"""

import numpy as np
import numpy.typing as npt
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FloatArray = npt.NDArray[np.float32]


def preprocess(image: Image.Image, image_size: int = IMAGE_SIZE) -> FloatArray:
    """Single image to a (3, H, W) float32 array, ImageNet-normalised."""
    resized = image.convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    gray = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    stacked = np.repeat(gray[None, :, :], 3, axis=0)
    return (stacked - mean) / std


def preprocess_batch(images: list[Image.Image], image_size: int = IMAGE_SIZE) -> FloatArray:
    if not images:
        raise ValueError("no images to preprocess")
    return np.stack([preprocess(image, image_size) for image in images])


def sigmoid(logits: npt.ArrayLike) -> FloatArray:
    # Clip before exp so a large negative logit does not overflow the float32
    # exponential; the result is already saturated at 0/1 there anyway.
    x = np.clip(np.asarray(logits, dtype=np.float32), -80.0, 80.0)
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float32)
