import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
import numpy.typing as npt
import typer

from castinspect import __version__
from castinspect.config import DEFAULT_CONFIG_PATH, Settings, load_settings
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

if TYPE_CHECKING:
    from castinspect.evaluation.robustness import LabelledImage

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
    save_folds: Annotated[
        bool, typer.Option(help="Save each fold's fitted model for later analyses.")
    ] = False,
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
        fold_model_dir=(settings.paths.artifacts / "models") if save_folds else None,
        fold_model_name=name,
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
        "castinspect.models.oneclass",
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


def _split_items(settings: Settings, root: Path, split: str) -> list["LabelledImage"]:
    from PIL import Image

    from castinspect.data.dataset import LABEL_TO_INDEX
    from castinspect.evaluation.robustness import LabelledImage

    rows = [r for r in read_manifest(settings.paths.data_processed / "manifest.csv")]
    items = []
    for row in rows:
        if row.split != split:
            continue
        # Dev images keep their fold (scored by the held-out model); test images
        # all map to fold 0, scored by the single final model.
        fold = row.fold if split == "dev" else 0
        with Image.open(root / row.path) as img:
            items.append(LabelledImage(fold, LABEL_TO_INDEX[row.label], img.convert("L")))
    return items


def _fold_models(settings: Settings, name: str = "efficientnet_b0") -> dict[int, Any]:
    from castinspect.models.efficientnet import EfficientNetClassifier

    models: dict[int, Any] = {}
    for fold in range(settings.split.n_folds):
        path = settings.paths.artifacts / "models" / f"{name}_fold{fold}.pt"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} missing; run castinspect cv configs/models/{name}.yaml --save-folds"
            )
        model = EfficientNetClassifier(pretrained=False)
        model.load(path)
        models[fold] = model
    return models


def _final_model(settings: Settings, name: str = "efficientnet_b0") -> dict[int, Any]:
    from castinspect.models.efficientnet import EfficientNetClassifier

    path = settings.paths.artifacts / "models" / f"{name}_final.pt"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing; run castinspect train-final")
    model = EfficientNetClassifier(pretrained=False)
    model.load(path)
    return {0: model}


def _policy_calibrator(
    policy_path: Path,
) -> tuple[Any, Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]]:
    from castinspect.decision.calibration import apply_temperature, probabilities_to_logits
    from castinspect.decision.policy import load_policy

    policy = load_policy(policy_path)

    def calibrate(probabilities: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return apply_temperature(probabilities_to_logits(probabilities), policy.temperature)

    return policy, calibrate


@app.command()
def robustness(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    perturbations: Annotated[Path, typer.Option(help="Perturbation config.")] = Path(
        "configs/robustness.yaml"
    ),
    policy_path: Annotated[Path, typer.Option(help="Frozen policy.")] = Path("configs/policy.yaml"),
    split: Annotated[str, typer.Option(help="dev (fold models) or test (final model).")] = "dev",
) -> None:
    """Perturbation suite under the frozen policy. On dev, each image is scored
    by the fold model that held it out; on test, by the final model."""
    from castinspect.evaluation.robustness import (
        load_perturbations,
        max_recall_drop,
        run_robustness,
    )

    settings = load_settings(config)
    root = locate_image_root(settings.paths.data_raw)
    specs, seed = load_perturbations(perturbations)
    policy, calibrate = _policy_calibrator(policy_path)
    models = (
        _fold_models(settings, policy.model)
        if split == "dev"
        else _final_model(settings, policy.model)
    )
    items = _split_items(settings, root, split)

    rows = run_robustness(
        lambda fold, images: models[fold].predict_images(images),
        items,
        specs,
        calibrate,
        policy.thresholds.review,
        policy.thresholds.reject,
        seed,
    )
    table = [
        {
            "perturbation": r.perturbation,
            "level": r.level,
            "severity": r.severity,
            "auroc": r.auroc,
            "defect_recall": r.outcome.defect_recall,
            "escapes": f"{r.outcome.escapes}/{r.outcome.n_defects}",
            "false_reject_rate": r.outcome.false_reject_rate,
            "false_rejects": f"{r.outcome.false_rejects}/{r.outcome.n_ok}",
            "review_share": r.outcome.review_share,
        }
        for r in rows
    ]
    worst, drop = max_recall_drop(rows, level=2)
    stem = "oof_robustness" if split == "dev" else "test_robustness"
    output = settings.paths.artifacts / "robustness" / f"{stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"rows": table, "max_medium_recall_drop": {"perturbation": worst, "points": drop}},
            indent=2,
        ),
        encoding="utf-8",
    )
    for entry in table:
        typer.echo(
            f"{entry['perturbation']:>15} L{entry['level']} sev={entry['severity']:<5} "
            f"auroc={entry['auroc']:.4f} recall={entry['defect_recall']:.4f} "
            f"escapes={entry['escapes']} false_rejects={entry['false_rejects']} "
            f"review={entry['review_share']:.3f}"
        )
    typer.echo(f"max recall drop at medium severity: {drop:.2f} points ({worst})")
    typer.echo(f"written to {output}")


@app.command()
def shortcut(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    policy_path: Annotated[Path, typer.Option(help="Frozen policy.")] = Path("configs/policy.yaml"),
    limit: Annotated[float, typer.Option(help="AUROC above this flags a shortcut.")] = 0.60,
) -> None:
    """AUROC from background alone: the part is masked out and each image is
    scored by its held-out fold model. Near 0.5 means no background shortcut."""
    from castinspect.evaluation.robustness import background_auroc

    settings = load_settings(config)
    root = locate_image_root(settings.paths.data_raw)
    policy, _ = _policy_calibrator(policy_path)
    models = _fold_models(settings, policy.model)
    items = _split_items(settings, root, "dev")
    value = background_auroc(lambda fold, images: models[fold].predict_images(images), items)

    output = settings.paths.artifacts / "robustness" / "shortcut.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"background_only_auroc": value, "n": len(items), "limit": limit}, indent=2),
        encoding="utf-8",
    )
    typer.echo(f"background-only AUROC: {value:.4f} over {len(items)} images (limit {limit})")
    if value > limit:
        typer.secho("possible background shortcut: review before continuing", err=True)
        raise typer.Exit(code=2)


@app.command()
def gradcam(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    policy_path: Annotated[Path, typer.Option(help="Frozen policy.")] = Path("configs/policy.yaml"),
    per_category: Annotated[int, typer.Option(help="Examples per category.")] = 3,
) -> None:
    """Grad-CAM grid over out-of-fold examples: caught defects, the hardest
    defects, false rejects and review-band parts. Written under artifacts/
    only: the dataset licence does not allow redistributing derived images."""
    from PIL import Image

    from castinspect.evaluation.cv import load_oof
    from castinspect.explain.gradcam import GradCam, overlay
    from castinspect.models.train import build_eval_transform

    settings = load_settings(config)
    root = locate_image_root(settings.paths.data_raw)
    policy, calibrate = _policy_calibrator(policy_path)
    models = _fold_models(settings, policy.model)
    oof = load_oof(settings.paths.artifacts / "oof" / f"{policy.model}.parquet")
    calibrated = calibrate(np.asarray([p.probability for p in oof]))
    review, reject = policy.thresholds.review, policy.thresholds.reject

    indexed = list(zip(oof, calibrated, strict=True))
    defects = sorted((x for x in indexed if x[0].label == 1), key=lambda x: x[1])
    healthy_rejected = sorted(
        (x for x in indexed if x[0].label == 0 and x[1] >= reject), key=lambda x: -x[1]
    )
    in_band = sorted((x for x in indexed if review <= x[1] < reject), key=lambda x: x[1])
    categories = {
        "caught defect": [x for x in defects if x[1] >= reject][-per_category:],
        "hardest defect": defects[:per_category],
        "false reject": healthy_rejected[:per_category],
        "review band": in_band[:per_category],
    }

    tile_size = 224
    tiles: list[npt.NDArray[np.uint8]] = []
    summary: dict[str, list[dict[str, object]]] = {}
    for category, examples in categories.items():
        summary[category] = []
        for prediction, probability in examples:
            with Image.open(root / prediction.path) as img:
                image = img.convert("L")
            model = models[prediction.fold]
            cam = GradCam(model.network)
            heatmap = cam.heatmap(build_eval_transform(model.image_size)(image))
            tile = overlay(image.resize((tile_size, tile_size)), heatmap)
            tiles.append(np.asarray(tile))
            summary[category].append(
                {
                    "path": prediction.path,
                    "label": prediction.label,
                    "probability": float(probability),
                }
            )
        while len(tiles) % per_category:
            tiles.append(np.full((tile_size, tile_size, 3), 255, dtype=np.uint8))

    rows = [np.hstack(tiles[i : i + per_category]) for i in range(0, len(tiles), per_category)]
    figure_dir = settings.paths.artifacts / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.vstack(rows)).save(figure_dir / "gradcam_oof_grid.png")
    (figure_dir / "gradcam_oof_grid.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    typer.echo(f"grid rows: {list(categories)}; written to {figure_dir / 'gradcam_oof_grid.png'}")


@app.command()
def efficiency(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    model_config: Annotated[Path, typer.Option(help="Supervised model YAML.")] = Path(
        "configs/models/efficientnet_b0.yaml"
    ),
    fractions: Annotated[str, typer.Option(help="Comma-separated defect fractions.")] = (
        "0.05,0.1,0.25,0.5,1.0"
    ),
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] = "0,1,2",
    epochs: Annotated[
        int | None, typer.Option(help="Fixed epochs; defaults to the CV median.")
    ] = None,
    oneclass_config: Annotated[
        Path, typer.Option(help="Embedding backbone settings for the one-class model.")
    ] = Path("configs/models/resnet18.yaml"),
) -> None:
    """Data-efficiency curve: AUROC versus share of defective training images,
    plus a one-class reference that sees no defect at all."""
    import statistics
    from dataclasses import asdict

    from castinspect.evaluation.cv import oof_arrays, run_cross_validation
    from castinspect.evaluation.efficiency import run_efficiency_point
    from castinspect.evaluation.intervals import bootstrap_auroc
    from castinspect.models.base import create_model

    settings = load_settings(config)
    root = locate_image_root(settings.paths.data_raw)
    manifest = read_manifest(settings.paths.data_processed / "manifest.csv")
    name, params, _ = _load_model_spec(model_config)
    if epochs is None:
        info = json.loads(
            (settings.paths.artifacts / "oof" / f"{name}_fit_info.json").read_text(encoding="utf-8")
        )
        epochs = int(statistics.median(fold["best_epoch"] for fold in info["folds"]))
    params["epochs_override"] = epochs
    typer.echo(f"supervised model {name}, fixed {epochs} epochs (no early stopping)")

    points = []
    for fraction in (float(f) for f in fractions.split(",")):
        for seed in (int(s) for s in seeds.split(",")):
            point = run_efficiency_point(
                lambda: create_model(name, **params),
                manifest,
                root,
                fraction,
                seed,
                settings.split.n_folds,
            )
            points.append(asdict(point))
            typer.echo(
                f"fraction {fraction:.2f} seed {seed} fold {point.eval_fold}: "
                f"{point.n_defects_train} defects, AUROC {point.auroc:.4f}"
            )

    # One-class reference on frozen embeddings: no defective image enters the
    # reference set and k is fixed a priori, since choosing it by AUROC would
    # need defect labels (ADR 0007). The curve's zero point stays honest.
    _, probe_params, _ = _load_model_spec(oneclass_config)
    oneclass_params: dict[str, object] = {
        "k": 5,
        "k_grid": (5,),
        **{
            key: probe_params[key]
            for key in ("pretrained", "weights_sha256", "cache_dir", "image_size")
            if key in probe_params
        },
    }
    result = run_cross_validation(
        lambda: create_model("oneclass_knn", **oneclass_params),
        manifest,
        root,
        settings.split.n_folds,
    )
    labels, scores, groups = oof_arrays(result.predictions)
    oneclass = bootstrap_auroc(labels, scores, groups)
    typer.echo(
        f"one-class kNN (k=5, zero defects): OOF AUROC {oneclass.point:.4f} "
        f"[{oneclass.lower:.4f}, {oneclass.upper:.4f}]"
    )

    output = settings.paths.artifacts / "efficiency" / "efficiency.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "model": name,
                "epochs": epochs,
                "points": points,
                "oneclass_knn": {"k": 5, "auroc": asdict(oneclass), "n": len(labels)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"written to {output}")


@app.command()
def evaluate(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    policy_path: Annotated[Path, typer.Option(help="Frozen policy.")] = Path("configs/policy.yaml"),
) -> None:
    """Final test-set evaluation, once, with the ONNX model and frozen policy.
    This is the only command that reads the test split."""
    import hashlib

    from PIL import Image

    from castinspect.data.acquire import sha256_file
    from castinspect.deploy.runtime import OnnxInspector
    from castinspect.evaluation.final import baseline_auroc, evaluate_test, metrics_to_dict

    settings = load_settings(config)
    policy, calibrate = _policy_calibrator(policy_path)
    root = locate_image_root(settings.paths.data_raw)
    manifest_path = settings.paths.data_processed / "manifest.csv"
    test_rows = [r for r in read_manifest(manifest_path) if r.split == "test"]
    typer.echo(f"TEST-SET EVALUATION (once): {len(test_rows)} images under frozen policy")

    onnx_path = settings.paths.artifacts / "onnx" / f"{policy.model}.onnx"
    inspector = OnnxInspector(onnx_path)
    images = []
    for row in test_rows:
        with Image.open(root / row.path) as img:
            images.append(img.convert("L"))
    labels = np.asarray([1 if r.label == "def" else 0 for r in test_rows], dtype=np.int64)
    groups = np.asarray([r.group_id for r in test_rows], dtype=np.int64)
    raw_probs = inspector.predict_proba(images)
    calibrated = calibrate(np.asarray(raw_probs, dtype=np.float64))

    metrics = evaluate_test(
        labels, calibrated, groups, policy.thresholds.review, policy.thresholds.reject
    )

    # Baselines on the same test images, for the "beats baseline" KPI.
    baselines: dict[str, float] = {}
    for name in ("hog", "resnet18_probe"):
        oof_path = settings.paths.artifacts / "oof" / f"{name}.parquet"
        if oof_path.is_file():
            baselines[name + "_oof_auroc"] = baseline_auroc(
                *_model_test_scores(settings, root, test_rows, name)
            )

    manifest_hash = sha256_file(manifest_path)
    model_hash = sha256_file(onnx_path)
    policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    results = {
        "evaluated_on": "test",
        "n_images": len(test_rows),
        "frozen_policy": {
            "model": policy.model,
            "temperature": policy.temperature,
            "thresholds": {
                "review": policy.thresholds.review,
                "reject": policy.thresholds.reject,
            },
            "version": policy.version,
        },
        "hashes": {
            "manifest_sha256": manifest_hash,
            "onnx_sha256": model_hash,
            "policy_sha256": policy_hash,
        },
        "metrics": metrics_to_dict(metrics),
        "baselines": baselines,
    }
    results_json = settings.paths.reports / "results.json"
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_results_markdown(settings.paths.reports / "results.md", results)

    m = metrics
    typer.echo(f"AUROC: {m.auroc.point:.4f} [{m.auroc.lower:.4f}, {m.auroc.upper:.4f}]")
    typer.echo(f"ECE: {m.ece:.4f}")
    typer.echo(
        f"defect recall: {m.defect_recall.point:.4f} ({m.defect_recall.raw}) "
        f"[{m.defect_recall.lower:.4f}, {m.defect_recall.upper:.4f}]"
    )
    typer.echo(
        f"escape rate: {m.escape_rate.point:.4f} ({m.escape_rate.raw}) "
        f"[{m.escape_rate.lower:.4f}, {m.escape_rate.upper:.4f}]"
    )
    typer.echo(
        f"false reject rate: {m.false_reject_rate.point:.4f} ({m.false_reject_rate.raw}) "
        f"[{m.false_reject_rate.lower:.4f}, {m.false_reject_rate.upper:.4f}]"
    )
    typer.echo(
        f"manual review rate: {m.manual_review_rate.point:.4f} ({m.manual_review_rate.raw}) "
        f"[{m.manual_review_rate.lower:.4f}, {m.manual_review_rate.upper:.4f}]"
    )
    typer.echo(f"results written to {results_json}")


def _model_test_scores(
    settings: Settings, root: Path, test_rows: list[ManifestRow], name: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Fit a baseline on development and score the test images, for the
    baseline comparison in the final report."""
    import importlib

    from castinspect.data.dataset import LABEL_TO_INDEX
    from castinspect.models.base import create_model

    for module in ("castinspect.models.hog", "castinspect.models.resnet"):
        importlib.import_module(module)
    dev_rows = [
        r for r in read_manifest(settings.paths.data_processed / "manifest.csv") if r.split == "dev"
    ]
    model = create_model(name)
    model.fit(
        [root / r.path for r in dev_rows],
        [LABEL_TO_INDEX[r.label] for r in dev_rows],
    )
    scores = model.predict_proba([root / r.path for r in test_rows])
    labels = np.asarray([LABEL_TO_INDEX[r.label] for r in test_rows], dtype=np.int64)
    return labels, np.asarray(scores, dtype=np.float64)


def _write_results_markdown(path: Path, results: dict[str, Any]) -> None:
    m = results["metrics"]

    def line(label: str, rate: dict[str, Any]) -> str:
        return (
            f"| {label} | {rate['point']:.4f} | {rate['raw']} "
            f"| [{rate['lower']:.4f}, {rate['upper']:.4f}] |"
        )

    auroc = m["auroc"]
    c = m["confusion"]
    lines = [
        "# Final test evaluation",
        "",
        f"Evaluated once on the {m['n']}-image test split "
        f"({m['n_defect']} defective, {m['n_ok']} ok) under the frozen policy.",
        "",
        "| metric | point | raw | 95% CI |",
        "|---|---|---|---|",
        f"| AUROC | {auroc['point']:.4f} | — | [{auroc['lower']:.4f}, {auroc['upper']:.4f}] |",
        f"| ECE | {m['ece']:.4f} | — | — |",
        line("defect recall", m["defect_recall"]),
        line("escape rate", m["escape_rate"]),
        line("false reject rate", m["false_reject_rate"]),
        line("manual review rate", m["manual_review_rate"]),
        "",
        "## Confusion by action",
        "",
        "| | ok | defect |",
        "|---|---|---|",
        f"| pass | {c['pass_ok']} | {c['pass_defect_escape']} (escape) |",
        f"| manual review | {c['review_ok']} | {c['review_defect']} |",
        f"| reject | {c['reject_ok_false']} (false) | {c['reject_defect']} |",
        "",
        "## Provenance",
        "",
        f"- manifest sha256: `{results['hashes']['manifest_sha256']}`",
        f"- onnx sha256: `{results['hashes']['onnx_sha256']}`",
        f"- policy sha256: `{results['hashes']['policy_sha256']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@app.command()
def export(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    model: Annotated[str, typer.Option(help="Model name of the final checkpoint.")] = (
        "efficientnet_b0"
    ),
    parity_images: Annotated[int, typer.Option(help="Images for the parity check.")] = 200,
) -> None:
    """Export the final model to ONNX and report PyTorch/ONNX parity."""
    from PIL import Image

    from castinspect.deploy.export_onnx import check_parity, export_to_onnx

    settings = load_settings(config)
    checkpoint = settings.paths.artifacts / "models" / f"{model}_final.pt"
    onnx_path = settings.paths.artifacts / "onnx" / f"{model}.onnx"
    # 224 is the model's input size (configs/models), not the dataset image size.
    export_to_onnx(checkpoint, onnx_path, 224)
    typer.echo(f"exported to {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    root = locate_image_root(settings.paths.data_raw)
    rows = [
        r for r in read_manifest(settings.paths.data_processed / "manifest.csv") if r.split == "dev"
    ]
    rng = np.random.default_rng(settings.seed)
    picked = rng.choice(len(rows), size=min(parity_images, len(rows)), replace=False)
    images = []
    for index in picked:
        with Image.open(root / rows[int(index)].path) as img:
            images.append(img.convert("L"))
    max_diff = check_parity(checkpoint, onnx_path, images)
    typer.echo(f"max |PyTorch - ONNX| probability over {len(images)} images: {max_diff:.2e}")
    if max_diff > 1e-4:
        typer.secho("parity worse than 1e-4", err=True)
        raise typer.Exit(code=1)


@app.command()
def figures(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    model: Annotated[str, typer.Option(help="Model name for the calibration figure.")] = (
        "efficientnet_b0"
    ),
) -> None:
    """Render the evaluation figures from the JSON artifacts produced by the
    calibrate, robustness and efficiency commands."""
    from castinspect.evaluation.report import (
        render_efficiency_curve,
        render_reliability,
        render_robustness_bars,
    )

    settings = load_settings(config)
    figure_dir = settings.paths.reports / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    render_efficiency_curve(
        settings.paths.artifacts / "efficiency" / "efficiency.json",
        figure_dir / "data_efficiency.png",
    )
    render_robustness_bars(
        settings.paths.artifacts / "robustness" / "oof_robustness.json",
        figure_dir / "robustness_oof.png",
    )
    render_reliability(
        settings.paths.artifacts / "calibration" / f"{model}_calibration.json",
        figure_dir / "reliability.png",
    )
    for name in ("data_efficiency.png", "robustness_oof.png", "reliability.png"):
        typer.echo(f"rendered {figure_dir / name}")


@app.command("label-sheet")
def label_sheet(
    config: ConfigOption = DEFAULT_CONFIG_PATH,
    count: Annotated[int, typer.Option(help="Number of defective images to list.")] = 100,
    output: Annotated[Path | None, typer.Option(help="CSV destination.")] = None,
) -> None:
    """Blank sheet of defective development images for a person to annotate
    with a visible defect type. The defect_type column is left empty: the
    dataset carries no defect types and none are inferred."""
    import csv

    settings = load_settings(config)
    rows = [
        r
        for r in read_manifest(settings.paths.data_processed / "manifest.csv")
        if r.split == "dev" and r.label == "def"
    ]
    rng = np.random.default_rng(settings.seed)
    picked = sorted(rng.choice(len(rows), size=min(count, len(rows)), replace=False))
    destination = output or settings.paths.artifacts / "labelling" / "defect_types.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "fold", "defect_type"])
        for index in picked:
            writer.writerow([rows[int(index)].path, rows[int(index)].fold, ""])
    typer.echo(f"{len(picked)} defective images listed in {destination}")


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
