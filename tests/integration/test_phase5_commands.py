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

SIZE = 48


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny synthetic project: two folds, trained fold checkpoints and a
    frozen policy, built through the real commands."""
    root = tmp_path_factory.mktemp("phase5")
    image_root = root / "raw" / "set"
    rng = np.random.default_rng(0)
    rows = []
    for i in range(24):
        label = "def" if i % 2 else "ok"
        folder = "def_front" if label == "def" else "ok_front"
        target = image_root / folder / f"img_{i}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        array = rng.integers(90, 160, size=(SIZE, SIZE)).astype(np.uint8)
        if label == "def":
            array[16:30, 16:30] = 255
        Image.fromarray(array, mode="L").save(target)
        rows.append(ManifestRow(f"{folder}/img_{i}.png", label, i, "dev", (i // 2) % 2))
    write_manifest(rows, root / "processed" / "manifest.csv")

    (root / "config.yaml").write_text(
        "\n".join(
            [
                "paths:",
                f"  data_raw: {(root / 'raw').as_posix()}",
                f"  data_processed: {(root / 'processed').as_posix()}",
                f"  artifacts: {(root / 'artifacts').as_posix()}",
                f"  reports: {(root / 'reports').as_posix()}",
                "split:",
                "  n_folds: 2",
            ]
        )
    )
    (root / "model.yaml").write_text(
        "\n".join(
            [
                "model: efficientnet_b0",
                "validation_early_stopping: true",
                "params:",
                "  pretrained: false",
                f"  image_size: {SIZE}",
                "  batch_size: 4",
                "  max_epochs: 2",
                "  patience: 1",
            ]
        )
    )
    (root / "probe.yaml").write_text(
        f"model: resnet18_probe\nparams:\n  pretrained: false\n  image_size: {SIZE}\n"
    )
    (root / "robustness.yaml").write_text(
        "perturbations:\n  brightness:\n    severities: [0.1, 0.25, 0.4]\nseed: 1\n"
    )

    config = str(root / "config.yaml")
    result = runner.invoke(
        app, ["cv", str(root / "model.yaml"), "--config", config, "--save-folds"]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(
        app, ["calibrate", "--config", config, "--policy-path", str(root / "policy.yaml")]
    )
    assert result.exit_code == 0, result.output
    return root


def test_robustness_command(workspace: Path) -> None:
    result = runner.invoke(
        app,
        [
            "robustness",
            "--config",
            str(workspace / "config.yaml"),
            "--perturbations",
            str(workspace / "robustness.yaml"),
            "--policy-path",
            str(workspace / "policy.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(
        (workspace / "artifacts" / "robustness" / "oof_robustness.json").read_text()
    )
    assert [row["level"] for row in report["rows"]] == [0, 1, 2, 3]
    assert report["max_medium_recall_drop"]["perturbation"] == "brightness"


def test_shortcut_command_reports_and_flags(workspace: Path) -> None:
    args = [
        "shortcut",
        "--config",
        str(workspace / "config.yaml"),
        "--policy-path",
        str(workspace / "policy.yaml"),
    ]
    relaxed = runner.invoke(app, [*args, "--limit", "1.0"])
    assert relaxed.exit_code == 0, relaxed.output
    value = json.loads((workspace / "artifacts" / "robustness" / "shortcut.json").read_text())
    assert 0.0 <= value["background_only_auroc"] <= 1.0
    strict = runner.invoke(app, [*args, "--limit", "-1.0"])
    assert strict.exit_code == 2


def test_gradcam_command(workspace: Path) -> None:
    result = runner.invoke(
        app,
        [
            "gradcam",
            "--config",
            str(workspace / "config.yaml"),
            "--policy-path",
            str(workspace / "policy.yaml"),
            "--per-category",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    grid = Image.open(workspace / "artifacts" / "figures" / "gradcam_oof_grid.png")
    assert grid.width == 224


def test_efficiency_command(workspace: Path) -> None:
    result = runner.invoke(
        app,
        [
            "efficiency",
            "--config",
            str(workspace / "config.yaml"),
            "--model-config",
            str(workspace / "model.yaml"),
            "--fractions",
            "0.5,1.0",
            "--seeds",
            "0",
            "--epochs",
            "1",
            "--oneclass-config",
            str(workspace / "probe.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((workspace / "artifacts" / "efficiency" / "efficiency.json").read_text())
    assert [p["fraction"] for p in report["points"]] == [0.5, 1.0]
    assert report["points"][0]["n_defects_train"] < report["points"][1]["n_defects_train"]
    assert 0.0 <= report["oneclass_knn"]["auroc"]["point"] <= 1.0


def test_label_sheet_lists_defects_with_blank_type(workspace: Path) -> None:
    import csv

    output = workspace / "sheet.csv"
    result = runner.invoke(
        app,
        [
            "label-sheet",
            "--config",
            str(workspace / "config.yaml"),
            "--count",
            "5",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    with output.open(newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    assert len(records) == 5
    assert all(r["path"].startswith("def_front/") for r in records)
    assert all(r["defect_type"] == "" for r in records)


def test_fold_models_missing_is_explicit(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  artifacts: {(tmp_path / 'a').as_posix()}\n")
    from castinspect.cli import _fold_models
    from castinspect.config import load_settings

    with pytest.raises(FileNotFoundError, match="--save-folds"):
        _fold_models(load_settings(config))
