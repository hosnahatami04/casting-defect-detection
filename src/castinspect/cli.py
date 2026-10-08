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
    read_manifest,
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
    root = image_root or locate_image_root(settings.paths.data_raw)
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


@app.command()
def cv(
    model_config: Annotated[Path, typer.Argument(help="Model YAML under configs/models/.")],
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    image_root: Annotated[
        Path | None, typer.Option(help="Dataset image root; defaults to a search in data_raw.")
    ] = None,
) -> None:
    from castinspect.evaluation.cv import oof_arrays, run_cross_validation, save_oof
    from castinspect.models.base import create_model

    settings = load_settings(config)
    root = image_root or locate_image_root(settings.paths.data_raw)
    manifest = read_manifest(settings.paths.data_processed / "manifest.csv")

    name, params, pass_validation = _load_model_spec(model_config)
    typer.echo(f"model: {name}; image root: {root}")

    result = run_cross_validation(
        lambda: create_model(name, **params),
        manifest,
        root,
        settings.split.n_folds,
        pass_validation=pass_validation,
    )
    oof_path = settings.paths.artifacts / "oof" / f"{name}.parquet"
    save_oof(result.predictions, oof_path)
    typer.echo(f"out-of-fold predictions written to {oof_path}")

    if any(result.fit_infos):
        info_path = settings.paths.artifacts / "oof" / f"{name}_fit_info.json"
        info = {"folds": result.fit_infos, "fold_seconds": result.fold_seconds}
        info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        typer.echo(f"fit details written to {info_path}")
    typer.echo(f"fold wall times (s): {[round(s) for s in result.fold_seconds]}")

    labels, probabilities, groups = oof_arrays(result.predictions)
    summary = _oof_summary(labels, probabilities, groups)
    metrics_path = settings.paths.artifacts / "oof" / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2))


def _load_model_spec(model_config: Path) -> tuple[str, dict[str, object], bool]:
    import importlib

    import yaml

    # Importing the model modules registers their factories.
    for module in (
        "castinspect.models.hog",
        "castinspect.models.resnet",
        "castinspect.models.efficientnet",
    ):
        importlib.import_module(module)

    spec = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    name = spec["model"]
    params = dict(spec.get("params") or {})
    if "cache_dir" in params and params["cache_dir"] is not None:
        params["cache_dir"] = Path(str(params["cache_dir"]))
    return name, params, bool(spec.get("validation_early_stopping", False))


@app.command("train-final")
def train_final(
    model_config: Annotated[Path, typer.Argument(help="Model YAML under configs/models/.")],
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    image_root: Annotated[
        Path | None, typer.Option(help="Dataset image root; defaults to a search in data_raw.")
    ] = None,
) -> None:
    """Train the final model on all development data, using the median
    early-stopping epoch from the cross-validation run."""
    import statistics
    import time

    from castinspect.data.dataset import LABEL_TO_INDEX
    from castinspect.models.base import create_model

    settings = load_settings(config)
    root = image_root or locate_image_root(settings.paths.data_raw)
    manifest = read_manifest(settings.paths.data_processed / "manifest.csv")
    dev_rows = [row for row in manifest if row.split == "dev"]

    name, params, _ = _load_model_spec(model_config)
    info_path = settings.paths.artifacts / "oof" / f"{name}_fit_info.json"
    fit_info = json.loads(info_path.read_text(encoding="utf-8"))
    best_epochs = [fold["best_epoch"] for fold in fit_info["folds"]]
    median_epochs = int(statistics.median(best_epochs))
    typer.echo(f"fold best epochs: {best_epochs}; training final model for {median_epochs}")

    params["epochs_override"] = median_epochs
    model = create_model(name, **params)
    started = time.perf_counter()
    model.fit(
        [root / row.path for row in dev_rows],
        [LABEL_TO_INDEX[row.label] for row in dev_rows],
    )
    elapsed = time.perf_counter() - started
    final_path = settings.paths.artifacts / "models" / f"{name}_final.pt"
    model.save(final_path)
    typer.echo(f"final model trained on {len(dev_rows)} images in {elapsed:.0f}s")
    typer.echo(f"saved to {final_path}")


def _oof_summary(
    labels: npt.NDArray[np.int64],
    probabilities: npt.NDArray[np.float64],
    groups: npt.NDArray[np.int64],
) -> dict[str, object]:
    from castinspect.evaluation.intervals import bootstrap_auroc, clopper_pearson
    from castinspect.evaluation.metrics import (
        confusion_at_threshold,
        fpr_at_recall,
        recall_at_fpr,
        threshold_at_recall,
    )

    auroc_ci = bootstrap_auroc(labels, probabilities, groups)
    recall_target = 0.99
    threshold = threshold_at_recall(labels, probabilities, recall_target)
    confusion = confusion_at_threshold(labels, probabilities, threshold)
    defects = confusion.true_positive + confusion.false_negative
    healthy = confusion.true_negative + confusion.false_positive
    caught = clopper_pearson(confusion.true_positive, defects)
    rejected_ok = clopper_pearson(confusion.false_positive, healthy)
    return {
        "n": len(labels),
        "n_defect": int(defects),
        "n_ok": int(healthy),
        "auroc": {"point": auroc_ci.point, "lower": auroc_ci.lower, "upper": auroc_ci.upper},
        "recall_at_5pct_fpr": recall_at_fpr(labels, probabilities, 0.05),
        "fpr_at_99pct_recall": fpr_at_recall(labels, probabilities, recall_target),
        "operating_point_at_99pct_recall": {
            "threshold": threshold,
            "defect_recall": {
                "point": caught.point,
                "lower": caught.lower,
                "upper": caught.upper,
                "raw": f"{confusion.true_positive}/{defects}",
            },
            "false_reject_rate": {
                "point": rejected_ok.point,
                "lower": rejected_ok.lower,
                "upper": rejected_ok.upper,
                "raw": f"{confusion.false_positive}/{healthy}",
            },
        },
    }


@app.command()
def calibrate(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    model: Annotated[str, typer.Option(help="Model whose OOF predictions to calibrate.")] = (
        "efficientnet_b0"
    ),
    policy_path: Annotated[Path, typer.Option(help="Where to freeze the policy.")] = Path(
        "configs/policy.yaml"
    ),
) -> None:
    """Fit calibration and freeze the decision policy, from out-of-fold
    predictions only; the test split is never read here."""
    import yaml

    from castinspect.decision.calibration import (
        apply_temperature,
        expected_calibration_error,
        fit_temperature,
        probabilities_to_logits,
        reliability_bins,
    )
    from castinspect.decision.cost import CostModel, best_two_way, three_way_cost
    from castinspect.decision.threshold import (
        cost_aware_review_threshold,
        operating_point,
        operating_point_table,
        reject_threshold,
    )
    from castinspect.evaluation.cv import load_oof, oof_arrays

    settings = load_settings(config)
    oof_path = settings.paths.artifacts / "oof" / f"{model}.parquet"
    labels, probabilities, _ = oof_arrays(load_oof(oof_path))
    logits = probabilities_to_logits(probabilities)

    temperature = fit_temperature(labels, logits)
    calibrated = apply_temperature(logits, temperature)
    ece_before = expected_calibration_error(labels, probabilities)
    ece_after = expected_calibration_error(labels, calibrated)
    typer.echo(f"temperature: {temperature:.4f}; ECE {ece_before:.4f} -> {ece_after:.4f}")

    costs = CostModel(
        escape=settings.decision.costs.escape,
        false_reject=settings.decision.costs.false_reject,
        manual_review=settings.decision.costs.manual_review,
    )
    recall_target = settings.decision.recall_target
    t_reject = reject_threshold(labels, calibrated, recall_target)
    t_review = cost_aware_review_threshold(
        labels, calibrated, t_reject, settings.decision.max_escape, costs
    )
    reject_point = operating_point(labels, calibrated, t_reject)
    review_point = operating_point(labels, calibrated, t_review)
    n_defects = int(labels.sum())
    escapes = n_defects - review_point.defects_caught
    reviews = int(np.sum((calibrated >= t_review) & (calibrated < t_reject)))
    typer.echo(
        f"t_reject={t_reject:.6f}: recall {reject_point.defect_recall:.4f} "
        f"({reject_point.defects_caught}/{n_defects}), "
        f"false rejects {reject_point.false_rejects}/{len(labels) - n_defects}"
    )
    typer.echo(
        f"t_review={t_review:.6f}: escapes {escapes}/{n_defects} "
        f"({escapes / n_defects:.4%}), review load {reviews}/{len(labels)} "
        f"({reviews / len(labels):.4%})"
    )

    frozen_cost = three_way_cost(labels, calibrated, t_review, t_reject, costs)
    best_threshold, best_cost = best_two_way(labels, calibrated, costs)
    typer.echo(
        f"expected cost per part: three-way {frozen_cost:.4f} vs "
        f"best two-way {best_cost:.4f} (threshold {best_threshold:.6f})"
    )

    table = operating_point_table(labels, calibrated, [0.95, 0.97, 0.98, 0.99, 0.995, 1.0])
    for point in table:
        typer.echo(
            f"  recall>={point.defect_recall:.4f} at t={point.threshold:.6f}: "
            f"false rejects {point.false_rejects} ({point.false_reject_rate:.2%})"
        )

    calibration_dir = settings.paths.artifacts / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    details = {
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "reliability_before": [vars(b) for b in reliability_bins(labels, probabilities)],
        "reliability_after": [vars(b) for b in reliability_bins(labels, calibrated)],
        "operating_points": [vars(p) for p in table],
        "frozen_cost_per_part": frozen_cost,
        "best_two_way": {"threshold": best_threshold, "cost_per_part": best_cost},
    }
    details_path = calibration_dir / f"{model}_calibration.json"
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    typer.echo(f"calibration details written to {details_path}")

    # Full precision on purpose: rounding a threshold can move it past a
    # borderline sample and silently break the recall guarantee.
    policy = {
        "version": "1",
        "model": model,
        "temperature": float(temperature),
        "thresholds": {"review": float(t_review), "reject": float(t_reject)},
        "recall_target": recall_target,
        "max_escape": settings.decision.max_escape,
        "oof_source": oof_path.as_posix(),
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
    typer.echo(f"policy frozen to {policy_path}")


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
