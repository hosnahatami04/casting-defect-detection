import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from castinspect.cli import app
from castinspect.data.split import ManifestRow, write_manifest

pytestmark = pytest.mark.ml

runner = CliRunner()

SIZE = 32


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(0)
    image_root = tmp_path / "images"
    rows = []
    index = 0
    for fold in range(2):
        for label, bright in (("ok", False), ("def", True)) * 4:
            rel = f"{'def_front' if label == 'def' else 'ok_front'}/img_{index}.png"
            target = image_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            array = rng.integers(0, 60, size=(SIZE, SIZE)).astype(np.uint8)
            if bright:
                array[8:20, 8:20] = 255
            Image.fromarray(array, mode="L").save(target)
            rows.append(ManifestRow(path=rel, label=label, group_id=index, split="dev", fold=fold))
            index += 1
    processed = tmp_path / "processed"
    write_manifest(rows, processed / "manifest.csv")

    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 42",
                f"image_size: {SIZE}",
                "paths:",
                f"  data_raw: {(tmp_path / 'raw').as_posix()}",
                f"  data_processed: {processed.as_posix()}",
                f"  artifacts: {(tmp_path / 'artifacts').as_posix()}",
                f"  reports: {(tmp_path / 'reports').as_posix()}",
                "split:",
                "  n_folds: 2",
            ]
        )
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text(
        "\n".join(
            [
                "model: hog",
                "params:",
                f"  image_size: {SIZE}",
                "  inner_cv_folds: 2",
            ]
        )
    )
    return config, model_config


def test_cv_command_end_to_end(tmp_path: Path) -> None:
    config, model_config = _build_dataset(tmp_path)
    image_root = tmp_path / "images"

    result = runner.invoke(
        app,
        ["cv", str(model_config), "--config", str(config), "--image-root", str(image_root)],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "artifacts" / "oof" / "hog.parquet").is_file()
    metrics = json.loads(
        (tmp_path / "artifacts" / "oof" / "hog_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["n"] == 16
    assert metrics["auroc"]["point"] > 0.9
    assert metrics["operating_point_at_99pct_recall"]["defect_recall"]["raw"] == "8/8"
