import io
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from castinspect.cli import app
from castinspect.data.acquire import sha256_file

runner = CliRunner()


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(32, 32)).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _write_archive(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True)
    archive = raw_dir / "dataset.zip"
    prefix = "casting_512x512/casting_512x512"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(f"{prefix}/ok_front/ok_0.png", _png_bytes(0))
        zf.writestr(f"{prefix}/ok_front/ok_1.png", _png_bytes(1))
        zf.writestr(f"{prefix}/def_front/def_0.png", _png_bytes(2))
    return archive


def _write_config(tmp_path: Path, archive_sha256: str = "") -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "paths:",
                f"  data_raw: {(tmp_path / 'raw').as_posix()}",
                f"  data_processed: {(tmp_path / 'processed').as_posix()}",
                f"  reports: {(tmp_path / 'reports').as_posix()}",
                "dataset:",
                f"  archive_sha256: '{archive_sha256}'",
            ]
        )
    )
    return config


def test_acquire_without_pin_reports_hash(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "raw")
    config = _write_config(tmp_path)
    result = runner.invoke(app, ["acquire", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert f"archive sha256: {sha256_file(archive)}" in result.output
    assert "ok_front: 2 files" in result.output
    assert "def_front: 1 files" in result.output


def test_acquire_with_pin_verifies(tmp_path: Path) -> None:
    archive = _write_archive(tmp_path / "raw")
    config = _write_config(tmp_path, archive_sha256=sha256_file(archive))
    result = runner.invoke(app, ["acquire", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "checksum verified" in result.output


def test_acquire_with_wrong_pin_fails(tmp_path: Path) -> None:
    _write_archive(tmp_path / "raw")
    config = _write_config(tmp_path, archive_sha256="0" * 64)
    result = runner.invoke(app, ["acquire", "--config", str(config)])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
