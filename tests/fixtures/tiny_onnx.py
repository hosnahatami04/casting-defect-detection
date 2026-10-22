"""A minimal ONNX model that mimics the real one's interface: input
(batch, 3, 224, 224), output a single logit per image. Built with the onnx
helper API so service and runtime tests need neither torch nor real weights.

The logit is the mean pixel of the first channel times a weight plus a bias,
so a bright image scores higher. Enough to exercise the full inference and
decision path deterministically.
"""

from pathlib import Path

import numpy as np


def write_tiny_onnx(path: Path, weight: float = 8.0, bias: float = -4.0) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path.parent.mkdir(parents=True, exist_ok=True)
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 224, 224])
    output_tensor = helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["batch"])

    w = numpy_helper.from_array(np.array([weight], dtype=np.float32), name="w")
    b = numpy_helper.from_array(np.array([bias], dtype=np.float32), name="b")

    # Mean over channels and spatial dims (axes as an attribute, valid at
    # opset 17), then an affine map to a single logit per image.
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Mul", ["mean", "w"], ["scaled"]),
        helper.make_node("Add", ["scaled", "b"], ["logit"]),
    ]
    graph = helper.make_graph(nodes, "tiny", [input_tensor], [output_tensor], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # onnx defaults to a newer IR version than onnxruntime accepts; pin to 10.
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path
