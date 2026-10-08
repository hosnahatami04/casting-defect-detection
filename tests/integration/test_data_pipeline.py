from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from castinspect.cli import app
from castinspect.data.split import read_manifest

runner = CliRunner()

SIZE = 32


def _write_images(root: Path, rng: np.random.Generator) -> None:
    ok_dir = root / "ok_front"
    def_dir = root / "def_front"
    ok_dir.mkdir(parents=True)
    def_dir.mkdir(parents=True)
    for i in range(30):
        array = rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8)
        Image.fromarray(array, mode="L").save(ok_dir / f"ok_{i}.png")
    for i in range(20):
        array = rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8)
        Image.fromarray(array, mode="L").save(def_dir / f"def_{i}.png")
    # One exact duplicate pair inside the same class to exercise grouping.
    base = rng.integers(0, 255, size=(SIZE, SIZE)).astype(np.uint8)
    Image.fromarray(base, mode="L").save(ok_dir / "dup_a.png")
    Image.fromarray(base, mode="L").save(ok_dir / "dup_b.png")


def _write_config(tmp_path: Path) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "seed: 42",
                f"image_size: {SIZE}",
                "paths:",
                f"  data_raw: {(tmp_path / 'raw').as_posix()}",
                f"  data_processed: {(tmp_path / 'processed').as_posix()}",
                f"  artifacts: {(tmp_path / 'artifacts').as_posix()}",
                f"  reports: {(tmp_path / 'reports').as_posix()}",
            ]
        )
    )
    return config


def test_data_command_end_to_end(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    image_root = tmp_path / "images"
    _write_images(image_root, rng)
    config = _write_config(tmp_path)

    result = runner.invoke(app, ["data", "--config", str(config), "--image-root", str(image_root)])

    assert result.exit_code == 0, result.output
    assert "no near-duplicate pair crosses a split or fold boundary" in result.output

    manifest = read_manifest(tmp_path / "processed" / "manifest.csv")
    assert len(manifest) == 52
    # The duplicate pair must share a group and therefore a section.
    dups = [r for r in manifest if r.path.startswith("ok_front/dup_")]
    assert len(dups) == 2
    assert dups[0].group_id == dups[1].group_id
    assert (dups[0].split, dups[0].fold) == (dups[1].split, dups[1].fold)

    assert (tmp_path / "processed" / "split_stats.json").is_file()
    assert (tmp_path / "reports" / "interim" / "phash_calibration.md").is_file()
