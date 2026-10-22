import os
from pathlib import Path

from fastapi import FastAPI

from castinspect.service.api import create_app


def app_from_env() -> FastAPI:
    """Entry point for uvicorn: castinspect.service.main:app_from_env, with
    paths and thread count taken from environment variables."""
    onnx_path = Path(os.environ.get("CASTINSPECT_ONNX", "artifacts/onnx/efficientnet_b0.onnx"))
    policy_path = Path(os.environ.get("CASTINSPECT_POLICY", "configs/policy.yaml"))
    threads = int(os.environ.get("CASTINSPECT_THREADS", "1"))
    return create_app(onnx_path, policy_path, threads=threads)
