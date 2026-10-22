import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

OPSET = 17


def export_to_onnx(checkpoint: Path, output: Path, image_size: int = 224) -> None:
    """Export a trained EfficientNet checkpoint to ONNX with a dynamic batch
    axis. The graph ends at the raw logit; sigmoid and the decision policy are
    applied outside, so the same ONNX file serves any downstream threshold."""
    import torch

    from castinspect.models.efficientnet import EfficientNetClassifier

    model = EfficientNetClassifier(pretrained=False)
    model.load(checkpoint)
    network = model.network
    network.eval()

    dummy = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The legacy TorchScript exporter keeps the dependency surface small (no
    # onnxscript) and is fully sufficient for a static CNN graph.
    torch.onnx.export(
        network,
        (dummy,),
        str(output),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )
    logger.info("exported %s to %s (opset %d)", checkpoint.name, output, OPSET)


def check_parity(
    checkpoint: Path, onnx_path: Path, images: list[Image.Image], image_size: int = 224
) -> float:
    """Maximum absolute difference in defect probability between PyTorch and
    ONNX over the given images."""
    from castinspect.deploy.preprocess import preprocess_batch, sigmoid
    from castinspect.deploy.runtime import OnnxInspector
    from castinspect.models.efficientnet import EfficientNetClassifier

    model = EfficientNetClassifier(pretrained=False)
    model.load(checkpoint)
    torch_probs = model.predict_images(images)

    inspector = OnnxInspector(onnx_path, image_size=image_size)
    batch = preprocess_batch(images, image_size)
    onnx_probs = sigmoid(inspector.logits(batch))
    return float(np.abs(torch_probs - onnx_probs).max())
