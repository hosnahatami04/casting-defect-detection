import json
import logging
from collections import Counter
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import typer

from castinspect import __version__
from castinspect.config import DEFAULT_CONFIG_PATH, load_settings
from castinspect.data.acquire import (
    find_archive,
    locate_image_root,
    safe_extract,
    verify_archive,
)
from castinspect.data.integrity import ImageRecord, scan_dataset
from castinspect.data.phash import (
    connected_groups,
    dhash_file,
    near_duplicate_pairs,
    pairwise_hamming,
)
from castinspect.data.split import (
    ManifestRow,
    SplitAssignment,
    conflicting_pairs,
    split_dataset,
    write_manifest,
)
from castinspect.logging import setup_logging

logger = logging.getLogger(__name__)

app = typer.Typer(name="castinspect", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option(help="Path to the YAML config file.")]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"castinspect {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    setup_logging()


@app.command()
def info(config: ConfigOption = DEFAULT_CONFIG_PATH) -> None:
    settings = load_settings(config)
    typer.echo(settings.model_dump_json(indent=2))


@app.command()
def acquire(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    archive: Annotated[
        Path | None, typer.Option(help="Dataset archive; defaults to the only zip in data/raw.")
    ] = None,
) -> None:
    settings = load_settings(config)
    archive_path = archive or find_archive(settings.paths.data_raw)
    actual = verify_archive(archive_path, settings.dataset.archive_sha256)
    if settings.dataset.archive_sha256:
        typer.echo(f"checksum verified: {actual}")
    else:
        typer.echo(f"archive sha256: {actual}")
        typer.echo("pin this value as dataset.archive_sha256 in configs/default.yaml")
    dest = settings.paths.data_raw / "extracted"
    safe_extract(archive_path, dest)
    root = locate_image_root(dest)
    typer.echo(f"image root: {root}")
    for class_dir in ("ok_front", "def_front"):
        count = sum(1 for p in (root / class_dir).iterdir() if p.is_file())
        typer.echo(f"{class_dir}: {count} files")


@app.command()
def data(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    image_root: Annotated[
        Path | None, typer.Option(help="Dataset image root; defaults to the extracted archive.")
    ] = None,
) -> None:
    settings = load_settings(config)
    root = image_root or locate_image_root(settings.paths.data_raw / "extracted")
    typer.echo(f"image root: {root}")

    report = scan_dataset(root, settings.image_size, frozenset(settings.dataset.allowed_modes))
    for rejected in report.rejected:
        typer.secho(f"rejected {rejected.path}: {rejected.reason}", err=True)
    typer.echo(f"accepted {len(report.accepted)} images, rejected {len(report.rejected)}")
    if not report.accepted:
        raise typer.Exit(code=1)

    records = report.accepted
    bits = np.stack([dhash_file(r.path) for r in records])
    distances = pairwise_hamming(bits)
    threshold = settings.grouping.dhash_threshold
    edges = near_duplicate_pairs(distances, threshold)
    group_ids = connected_groups(len(records), [(i, j) for i, j, _ in edges])
    n_groups = len(set(group_ids))
    typer.echo(f"{len(edges)} near-duplicate pairs at threshold {threshold}; {n_groups} groups")

    labels = [r.label for r in records]
    assignments = split_dataset(
        labels,
        group_ids,
        settings.split.test_fraction,
        settings.split.n_folds,
        settings.seed,
    )

    # Split and fold leakage in one pass: any close pair must share split and fold.
    section = [f"{a.split}:{a.fold}" for a in assignments]
    leaks = conflicting_pairs(distances, threshold, section)
    if leaks:
        for i, j, dist in leaks:
            typer.secho(
                f"leak: {records[i].path} ({section[i]}) ~ {records[j].path} "
                f"({section[j]}) at distance {dist}",
                err=True,
            )
        raise typer.Exit(code=1)
    typer.echo("no near-duplicate pair crosses a split or fold boundary")

    rows = [
        ManifestRow(
            path=record.path.relative_to(root).as_posix(),
            label=record.label,
            group_id=group,
            split=assignment.split,
            fold=assignment.fold,
        )
        for record, group, assignment in zip(records, group_ids, assignments, strict=True)
    ]
    manifest_path = settings.paths.data_processed / "manifest.csv"
    write_manifest(rows, manifest_path)
    typer.echo(f"manifest written to {manifest_path}")

    stats = _split_stats(records, assignments, settings.split.n_folds)
    stats_path = settings.paths.data_processed / "split_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    typer.echo(json.dumps(stats, indent=2))

    calibration_path = settings.paths.reports / "interim" / "phash_calibration.md"
    _write_calibration_report(calibration_path, records, distances, group_ids, threshold)
    typer.echo(f"threshold calibration report written to {calibration_path}")


def _split_stats(
    records: list[ImageRecord],
    assignments: list[SplitAssignment],
    n_folds: int,
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    sections = ["test"] + [f"fold_{f}" for f in range(n_folds)]
    for name in sections:
        stats[name] = {"ok": 0, "def": 0}
    for record, assignment in zip(records, assignments, strict=True):
        name = "test" if assignment.split == "test" else f"fold_{assignment.fold}"
        stats[name][record.label] += 1
    return stats


def _write_calibration_report(
    path: Path,
    records: list[ImageRecord],
    distances: npt.NDArray[np.uint8],
    group_ids: list[int],
    threshold: int,
) -> None:
    n = len(records)
    upper = distances[np.triu_indices(n, k=1)]
    histogram = {int(d): int((upper == d).sum()) for d in range(0, 25) if int((upper == d).sum())}
    group_sizes = Counter(Counter(group_ids).values())

    borderline: list[str] = []
    close = np.triu((distances > threshold) & (distances <= threshold + 4), k=1)
    rows, cols = np.nonzero(close)
    for i, j in list(zip(rows, cols, strict=True))[:20]:
        borderline.append(
            f"| {records[int(i)].path.name} | {records[int(j)].path.name} "
            f"| {int(distances[i, j])} |"
        )

    lines = [
        "# Perceptual hash threshold calibration",
        "",
        f"Pairwise dHash Hamming distances over {n} images "
        f"({n * (n - 1) // 2} pairs); current threshold: {threshold}.",
        "",
        "## Distance histogram (distances 0 to 24)",
        "",
        "| distance | pairs |",
        "|---|---|",
        *[f"| {d} | {c} |" for d, c in sorted(histogram.items())],
        "",
        "## Group size distribution",
        "",
        "| group size | groups |",
        "|---|---|",
        *[f"| {size} | {count} |" for size, count in sorted(group_sizes.items())],
        "",
        f"## Borderline pairs just above the threshold ({threshold + 1} to {threshold + 4})",
        "",
        "These pairs are NOT grouped. Review a sample to confirm the threshold is not too low.",
        "",
        "| image a | image b | distance |",
        "|---|---|---|",
        *borderline,
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
