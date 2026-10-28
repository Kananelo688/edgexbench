#!/usr/bin/env python3
"""
FILENAME:       src/edgebench/train/classification.py
DESCRIPTION:    Command-line entry point for classification training. Builds the decoded-image
                caches if absent, then trains every model in the experiment that is not already
                finished under the current recipe. This is just entery-point the training logic lives in
                train/trainer.py and the policy in core/.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edgebench.core.artifacts import CheckpointMeta, save_checkpoint
from edgebench.core.config import ConfigError, ExperimentConfig, RecipeConfig, RunConfig
from edgebench.core.data.cache import (CacheError, ImageCache, build_cache, cache_exists,
                                       discover_class_folders)
from edgebench.core.paths import Paths, ensure_dir, get_paths
from edgebench.core.registry import ModelSpec, get_registry
from edgebench.core.results import (STATUS_COMPLETED, STATUS_FAILED, ResultsLayout,
                                    ResultsStore)
from edgebench.train.trainer import EpochRecord, Trainer, TrainingResult

TASK = "classification"

# Sub-directories of the dataset root used for each split. Overridable from the command line
# for datasets laid out differently.
DEFAULT_SPLIT_DIRS: dict[str, list[str]] = {
    "train": ["train.X1", "train.X2", "train.X3", "train.X4"],
    "val": ["val.X"],
}


def classification_transform(resolution: int, resize_to: int):
    """Build the preprocessing applied before caching.

    Short-side resize followed by a centre crop, the standard ImageNet protocol. The function
    is passed to the cache builder rather than hard-coded there, so other tasks supply their
    own geometry without changing the cache machinery.

    Args:
        resolution: Output side length.
        resize_to: Short-side resize applied before cropping.

    Returns:
        Callable mapping a PIL image to an HWC uint8 array.
    """
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(resize_to), T.CenterCrop(resolution)])
    return lambda img: np.asarray(tf(img), dtype=np.uint8)


def ensure_caches(paths: Paths, dataset: str, recipe: RecipeConfig,
                  split_dirs: dict[str, list[str]], *, rebuild: bool = False) -> None:
    """Build the training and validation caches if they are not already present.

    Training is built first because it defines the canonical class ordering; validation is then
    built against that list, so a label index means the same class in both splits.

    Args:
        paths: Resolved project layout.
        dataset: Dataset identifier.
        recipe: Recipe supplying the two resolutions.
        split_dirs: Sub-directories of the dataset root for each split.
        rebuild: Rebuild even when a cache is present.

    Returns:
        None.

    Raises:
        CacheError: If the dataset tree is missing or inconsistent.
    """
    root = paths.dataset_dir(dataset)
    cache_dir = ensure_dir(paths.cache_dir(dataset))

    train_needed = rebuild or not cache_exists(cache_dir, "train", recipe.train_res)
    val_needed = rebuild or not cache_exists(cache_dir, "val", recipe.eval_res)
    if not (train_needed or val_needed):
        return

    classes: list[str] | None = None
    if not train_needed:
        classes = ImageCache.open(cache_dir, "train", recipe.train_res).classes

    if train_needed:
        found = discover_class_folders([root / d for d in split_dirs["train"]])
        print(f"  discovered {len(found)} train images in {len(found.classes)} classes")
        build_cache(found.samples,
                    classification_transform(recipe.train_res, recipe.train_res),
                    cache_dir, "train", recipe.train_res,
                    dataset=dataset, classes=found.classes,
                    transform_description=(f"Resize({recipe.train_res})"
                                           f"->CenterCrop({recipe.train_res})"),
                    source_roots=found.roots, overwrite=rebuild)
        classes = found.classes

    if val_needed:
        # The training class list is passed through, so a class present in one split and not
        # the other fails loudly instead of shifting every label index.
        found = discover_class_folders([root / d for d in split_dirs["val"]], classes=classes)
        print(f"  discovered {len(found)} val images in {len(found.classes)} classes")
        build_cache(found.samples,
                    classification_transform(recipe.eval_res, recipe.eval_resize),
                    cache_dir, "val", recipe.eval_res,
                    dataset=dataset, classes=found.classes,
                    transform_description=(f"Resize({recipe.eval_resize})"
                                           f"->CenterCrop({recipe.eval_res})"),
                    source_roots=found.roots, overwrite=rebuild)


def make_store(paths: Paths, debug: bool) -> ResultsStore:
    """Construct the results store, isolating debug runs.

    Debug output is routed to separate directories so a smoke test cannot contaminate recorded
    results or cause a real model to be skipped.

    Args:
        paths: Resolved project layout.
        debug: Whether this is a debug run.

    Returns:
        The configured store.
    """
    suffix = "_debug" if debug else ""
    results_dir = ensure_dir(paths.results_dir("training", TASK + suffix))
    artifact_dir = ensure_dir(paths.checkpoint_dir("pytorch", TASK + suffix))
    return ResultsStore(ResultsLayout(results_dir=results_dir, artifact_dir=artifact_dir))


def summary_row(spec: ModelSpec, result: TrainingResult, recipe: RecipeConfig,
                run: RunConfig, dataset: str, checkpoint: Path, paths: Paths) -> dict[str, Any]:
    """Assemble the summary table row for a completed model.

    Args:
        spec: Architecture that was trained.
        result: Training outcome.
        recipe: Recipe used.
        run: Execution settings used.
        dataset: Dataset identifier.
        checkpoint: Path of the saved checkpoint.
        paths: Resolved project layout, used to record a portable relative path.

    Returns:
        Mapping of column name to value.
    """
    import torch
    final = result.final
    return {
        "model": spec.name,
        "status": STATUS_COMPLETED,
        "recipe_digest": recipe.digest(),
        "source": result.resolved.source,
        "provider_version": result.resolved.provider_version,
        "dataset": dataset,
        "epochs": recipe.epochs,
        # The reported accuracy is the FINAL epoch, not the best. Selecting the best epoch on
        # the same split that is reported would be selecting on the test set.
        "top1": final.top1,
        "top5": final.top5,
        "val_loss": final.val_loss,
        "train_loss": final.train_loss,
        "best_top1": result.best_top1,
        "best_epoch": result.best_epoch,
        "train_res": recipe.train_res,
        "eval_res": recipe.eval_res,
        "batch_size": recipe.batch_size,
        "micro_batch_size": run.micro_batch_size or recipe.batch_size,
        "train_minutes": result.train_minutes,
        "seconds_per_epoch": result.seconds_per_epoch,
        "images_per_second": final.images_per_second,
        "peak_memory_mb": result.peak_memory_mb,
        "seed": recipe.seed,
        "amp_dtype": recipe.amp_dtype,
        "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                   and run.device != "cpu" else "cpu"),
        "torch_version": torch.__version__,
        "checkpoint": str(paths.relative(checkpoint)),
        "error": "",
    }


def train_one(spec: ModelSpec, exp: ExperimentConfig, paths: Paths, store: ResultsStore,
              train_cache: ImageCache, val_cache: ImageCache) -> dict[str, Any]:
    """Train a single model and persist its checkpoint and metrics.

    Args:
        spec: Architecture to train.
        exp: Experiment configuration.
        paths: Resolved project layout.
        store: Results store.
        train_cache: Training image cache.
        val_cache: Validation image cache.

    Returns:
        The summary row describing the outcome.
    """
    recipe, run = exp.recipe, exp.run
    final_path = store.artifact_path(spec.name, ".pt")
    resume_path = store.layout.artifact_dir / f"{spec.name}.resume.pt"

    # Epoch rows beyond the resume point are discarded first: a crash between logging an epoch
    # and writing its checkpoint would otherwise leave a duplicate epoch in the curve.
    resume_epoch = 0
    if resume_path.exists() and run.resume:
        from edgebench.core.artifacts import load_checkpoint
        try:
            ts = load_checkpoint(resume_path).get("train_state") or {}
            resume_epoch = int(ts.get("epoch", 0))
        except Exception:
            resume_epoch = 0
    store.truncate_epochs(spec.name, after_epoch=resume_epoch)

    def record_epoch(rec: EpochRecord) -> None:
        store.log_epoch(rec.as_row(model=spec.name, recipe_digest=recipe.digest()))
        print(f"    epoch {rec.epoch:3d}/{recipe.epochs}  loss {rec.train_loss:.3f}  "
              f"top1 {rec.top1:6.2f}  top5 {rec.top5:6.2f}  "
              f"{rec.epoch_seconds:.0f}s  {rec.images_per_second:.0f} img/s")

    trainer = Trainer(spec, recipe, run, train_cache, val_cache,
                      resume_path=resume_path, on_epoch=record_epoch)
    result = trainer.fit()

    meta = CheckpointMeta(
        name=spec.name, source=result.resolved.source,
        provider_version=result.resolved.provider_version,
        kwargs=dict(spec.kwargs), num_classes=recipe.num_classes,
        classes=list(train_cache.classes), recipe_digest=recipe.digest(),
        task=exp.task, dataset=exp.dataset, epochs_completed=result.epochs_completed,
        metrics={"top1": result.final.top1, "top5": result.final.top5,
                 "val_loss": result.final.val_loss},
        extra={"best_top1": result.best_top1, "best_epoch": result.best_epoch,
               "also_available_in": list(result.resolved.also_available_in)},
    )
    save_checkpoint(final_path, result.resolved.module, meta, recipe)
    # The resume file is scratch; removing it on success keeps the checkpoint directory free of
    # files that look like deliverables but are not.
    resume_path.unlink(missing_ok=True)

    row = summary_row(spec, result, recipe, run, exp.dataset, final_path, paths)
    store.write_sidecar(final_path, {**meta.to_dict(), "recipe": recipe.to_dict()})
    return row


def build_experiment(args: argparse.Namespace) -> ExperimentConfig:
    """Assemble the experiment configuration from a file and command-line overrides.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The experiment configuration.

    Raises:
        ConfigError: If the file is invalid or no models are specified.
    """
    exp = (ExperimentConfig.from_yaml(args.config) if args.config
           else ExperimentConfig(task=TASK, recipe=RecipeConfig(), run=RunConfig()))
    exp.task = TASK
    if args.dataset:
        exp.dataset = args.dataset
    if args.models:
        exp.models = list(args.models)

    recipe_overrides = {k: v for k, v in
                        (("epochs", args.epochs), ("batch_size", args.batch_size),
                         ("lr", args.lr), ("seed", args.seed),
                         ("num_classes", args.num_classes),
                         ("warmup_epochs", args.warmup_epochs))
                        if v is not None}
    if recipe_overrides:
        exp.recipe = exp.recipe.derive(**recipe_overrides)

    if args.micro_batch_size is not None:
        exp.run.micro_batch_size = args.micro_batch_size
    if args.device:
        exp.run.device = args.device
    if args.limit_batches is not None:
        exp.run.limit_batches = args.limit_batches
    if args.debug:
        exp.run.debug = True
    if args.no_resume:
        exp.run.resume = False
    if args.compile:
        exp.run.compile = True

    if not exp.models:
        raise ConfigError("no models specified; use --models or a config file")
    return exp


def main(argv: Sequence[str] | None = None) -> int:
    """Run the classification training sweep.

    Args:
        argv: Command-line arguments. None reads sys.argv.

    Returns:
        Process exit status: 0 when every model completed, 1 otherwise.
    """
    ap = argparse.ArgumentParser(
        description="Train classification models from scratch under one shared recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", type=Path, help="experiment YAML")
    ap.add_argument("--models", nargs="*", help="model names, overriding the config")
    ap.add_argument("--dataset", help="dataset identifier under datasets/")
    ap.add_argument("--only", nargs="*", help="restrict the run to these models")
    ap.add_argument("--force", nargs="*", default=[],
                    help="retrain these models even if already completed")
    ap.add_argument("--force-all", action="store_true", help="retrain everything")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--micro-batch-size", type=int,
                    help="per-step batch when the effective batch does not fit")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--warmup-epochs", type=int,
                    help="must be below --epochs; lower it for short runs")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--num-classes", type=int)
    ap.add_argument("--device", help="cuda, cpu, or auto")
    ap.add_argument("--compile", action="store_true", help="wrap the model with torch.compile")
    ap.add_argument("--limit-batches", type=int, help="cap batches per epoch (smoke tests)")
    ap.add_argument("--debug", action="store_true",
                    help="write to isolated debug directories")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--cache-only", action="store_true",
                    help="build the caches and exit without training")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be trained and exit")
    ap.add_argument("--split-dir", nargs="*", metavar="SPLIT=DIR1,DIR2",
                    help="override split directories, e.g. train=train.X1,train.X2")
    args = ap.parse_args(argv)

    try:
        exp = build_experiment(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    split_dirs = dict(DEFAULT_SPLIT_DIRS)
    for item in args.split_dir or []:
        split, _, dirs = item.partition("=")
        split_dirs[split] = dirs.split(",")

    paths = get_paths()
    recipe, run = exp.recipe, exp.run
    print(f"recipe  {recipe.summary()}")
    print(f"dataset {exp.dataset}   task {exp.task}   debug={run.debug}")

    try:
        ensure_caches(paths, exp.dataset, recipe, split_dirs, rebuild=args.rebuild_cache)
    except CacheError as exc:
        print(f"cache error: {exc}", file=sys.stderr)
        return 2
    if args.cache_only:
        print("caches ready; --cache-only requested, stopping here")
        return 0

    cache_dir = paths.cache_dir(exp.dataset)
    train_cache = ImageCache.open(cache_dir, "train", recipe.train_res)
    val_cache = ImageCache.open(cache_dir, "val", recipe.eval_res)
    print(f"  {train_cache.describe()}")
    print(f"  {val_cache.describe()}")

    specs = [ModelSpec.coerce(m) for m in exp.models]
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
    names = [s.name for s in specs]
    force = names if args.force_all else args.force

    store = make_store(paths, run.debug)
    todo, done = store.pending(names, recipe.digest(), force=force, suffix=".pt")
    print(f"\nalready trained ({len(done)}): {done}")
    print(f"to train       ({len(todo)}): {todo}")
    if args.dry_run:
        return 0

    by_name = {s.name: s for s in specs}
    failures: list[str] = []

    for i, name in enumerate(todo, 1):
        spec = by_name[name]
        print(f"\n=== [{i}/{len(todo)}] {name} ===")
        try:
            row = train_one(spec, exp, paths, store, train_cache, val_cache)
            print(f"  done: top1 {row['top1']:.2f}  top5 {row['top5']:.2f}  "
                  f"{row['train_minutes']:.1f} min")
        except Exception as exc:  # one model must not end the sweep
            message = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {message}", file=sys.stderr)
            traceback.print_exc()
            failures.append(name)
            row = {"model": name, "status": STATUS_FAILED,
                   "recipe_digest": recipe.digest(), "error": message[:400]}
        store.record(row)

    digests = store.digests()
    if len(digests) > 1:
        print(f"\nWARNING: the summary table mixes {len(digests)} recipe digests {digests}. "
              f"Rows from different recipes are not directly comparable.")

    print(f"\ntrained {len(todo) - len(failures)}/{len(todo)}; "
          f"results in {store.layout.summary_path}")
    if failures:
        print(f"failed: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())