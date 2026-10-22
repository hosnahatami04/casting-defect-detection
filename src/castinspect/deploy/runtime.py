from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from castinspect.deploy.preprocess import preprocess_batch, sigmoid

FloatArray = npt.NDArray[np.float32]


class OnnxInspector:
    """ONNX Runtime inference session on CPU with a configurable thread count.

    Shares preprocessing with every other inference path, so a prediction here
    matches a prediction through the library or the API bit for bit.
    """

    def __init__(self, onnx_path: Path, image_size: int = 224, threads: int = 1) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._image_size = image_size

    def logits(self, batch: FloatArray) -> FloatArray:
        outputs = self._session.run(None, {self._input_name: batch.astype(np.float32)})
        return np.asarray(outputs[0], dtype=np.float32).reshape(-1)

    def predict_proba(self, images: list[Image.Image]) -> FloatArray:
        batch = preprocess_batch(images, self._image_size)
        return sigmoid(self.logits(batch))

    def predict_one(self, image: Image.Image) -> float:
        return float(self.predict_proba([image])[0])
