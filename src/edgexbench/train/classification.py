#!/usr/bin/env python3
"""
FILENAME:       src/edgexbench/train/classification.py
DESCRIPTION:    Command-line entry point for classification training. Builds the decoded-image
                caches if absent, then trains every model in the experiment that is not already
                finished under the current recipe. This is just entery-point the training logic lives in
                train/trainer.py and the policy in core/.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from edgexbench.core.artifacts import CheckpointMeta, load_checkpoint, save_checkpoint
from edgexbench.core.config import ConfigError, ExperimentConfig, RecipeConfig, RunConfig
from edgexbench.core.data.cache import (LABEL_FROM, CacheError, ImageCache, build_cache,
                                       cache_exists, discover_class_folders)
from edgexbench.core.paths import PathError, Paths, ensure_dir, get_paths
from edgexbench.core.registry import ModelSpec, RegistryError
from edgexbench.core.results import (STATUS_COMPLETED, STATUS_FAILED, ResultsError,
                                    ResultsLayout, ResultsStore)
from edgexbench.train.trainer import EpochRecord, Trainer, TrainingResult, resolve_device
from edgexbench.models.external import register_all

register_all()

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
                  split_dirs: dict[str, list[str]], *, rebuild: bool = False,
                  label_from: str = "parent", workers: int | None = None) -> None:
    """Build the training and validation caches if they are not already present.

    Training is built first because it defines the canonical class ordering; validation is then
    built against that list, so a label index means the same class in both splits.

    Args:
        paths: Resolved project layout.
        dataset: Dataset identifier.
        recipe: Recipe supplying the resolutions and the expected class count.
        split_dirs: Sub-directories of the dataset root for each split.
        rebuild: Rebuild even when a cache is present.
        label_from: How labels are read from paths; see cache.LABEL_FROM. "parent" suits the
            Kaggle ImageNet-100 layout.
        workers: Decoding threads; None uses the cache builder's default.

    Returns:
        None.

    Raises:
        CacheError: If the dataset tree is missing or inconsistent, or the number of classes
            found differs from recipe.num_classes.
    """
    root = paths.dataset_dir(dataset)
    cache_dir = ensure_dir(paths.cache_dir(dataset))
    # [AUDIT P1] The training cache is stored at train_cache_resolution (default
    # 1.25 x train_res), not train_res, so random crops are drawn from more detail than they
    # output instead of being upsampled from a 160px centre crop. The square centre crop itself
    # remains, as a fixed-shape cache requires it.
    # Before:
    #   train_needed = rebuild or not cache_exists(cache_dir, "train", recipe.train_res)
    #   ...
    #   classification_transform(recipe.train_res, recipe.train_res),
    #   cache_dir, "train", recipe.train_res,
    train_res = recipe.train_cache_resolution

    train_needed = rebuild or not cache_exists(cache_dir, "train", train_res)
    val_needed = rebuild or not cache_exists(cache_dir, "val", recipe.eval_res)
    if not (train_needed or val_needed):
        return

    classes: list[str] | None = None
    if not train_needed:
        classes = ImageCache.open(cache_dir, "train", train_res).classes

    if train_needed:
        found = discover_class_folders([root / d for d in split_dirs["train"]],
                                       label_from=label_from)
        print(f"  discovered {len(found)} train images in {len(found.classes)} classes")
        # [AUDIT C1] Labels are read from the path by a convention the caller picks, so a
        # wrong pick (or a stray layout) must fail here rather than train on, say, 4 classes
        # named after the part directories.
        # Before: (no check)
        if len(found.classes) != recipe.num_classes:
            raise CacheError(
                f"found {len(found.classes)} train classes but the recipe expects "
                f"{recipe.num_classes} (label_from={label_from!r}; first classes: "
                f"{found.classes[:5]}). Check --split-dir and --label-from.")
        build_cache(found.samples,
                    classification_transform(train_res, train_res),
                    cache_dir, "train", train_res,
                    dataset=dataset, classes=found.classes,
                    transform_description=(f"Resize({train_res})"
                                           f"->CenterCrop({train_res})"),
                    source_roots=found.roots, workers=workers, overwrite=rebuild)
        classes = found.classes

    if val_needed:
        # The training class list is passed through, so a class present in one split and not
        # the other fails loudly instead of shifting every label index.
        found = discover_class_folders([root / d for d in split_dirs["val"]], classes=classes,
                                       label_from=label_from)
        print(f"  discovered {len(found)} val images in {len(found.classes)} classes")
        build_cache(found.samples,
                    classification_transform(recipe.eval_res, recipe.eval_resize),
                    cache_dir, "val", recipe.eval_res,
                    dataset=dataset, classes=found.classes,
                    transform_description=(f"Resize({recipe.eval_resize})"
                                           f"->CenterCrop({recipe.eval_res})"),
                    source_roots=found.roots, workers=workers, overwrite=rebuild)


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
    # [AUDIT P6] The device column reported GPU 0's name whenever CUDA existed, even for
    # --device cuda:1. It now names the device the trainer actually resolved.
    # Before:
    #   "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
    #              and run.device != "cpu" else "cpu"),
    device = resolve_device(run.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else device.type
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
        "train_cache_res": recipe.train_cache_resolution,
        "eval_res": recipe.eval_res,
        "batch_size": recipe.batch_size,
        "micro_batch_size": run.micro_batch_size or recipe.batch_size,
        "train_minutes": result.train_minutes,
        "seconds_per_epoch": result.seconds_per_epoch,
        "images_per_second": final.images_per_second,
        "peak_memory_mb": result.peak_memory_mb,
        "seed": recipe.seed,
        "amp_dtype": recipe.amp_dtype,
        "device": device_name,
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
        The summary row describing the outcome, already recorded in the store.
    """
    recipe, run = exp.recipe, exp.run
    digest = recipe.digest()
    final_path = store.artifact_path(spec.name, ".pt")
    # [AUDIT P6] Built through artifact_path so the name is made filename-safe.
    # Before:
    #   resume_path = store.layout.artifact_dir / f"{spec.name}.resume.pt"
    resume_path = store.artifact_path(spec.name, ".resume.pt")

    # Epoch rows beyond the resume point are discarded first: a crash between logging an epoch
    # and writing its checkpoint would otherwise leave a duplicate epoch in the curve.
    # [AUDIT P2] The resume epoch was trusted without checking the checkpoint's recipe. A
    # resume file from another recipe is ignored by the trainer (restart at epoch 0), yet its
    # epoch kept rows 1..k alive, duplicating the new run's epochs. Truncation is now keyed on
    # (model, digest) and uses the resume epoch only when the digests match.
    # Before:
    #   ts = load_checkpoint(resume_path).get("train_state") or {}
    #   resume_epoch = int(ts.get("epoch", 0))
    #   ...
    #   store.truncate_epochs(spec.name, after_epoch=resume_epoch)
    resume_epoch = 0
    if resume_path.exists() and run.resume:
        try:
            payload = load_checkpoint(resume_path)
            if (payload.get("meta") or {}).get("recipe_digest") == digest:
                resume_epoch = int((payload.get("train_state") or {}).get("epoch", 0))
        except Exception:
            resume_epoch = 0
    store.truncate_epochs(spec.name, after_epoch=resume_epoch, digest=digest)

    def record_epoch(rec: EpochRecord) -> None:
        store.log_epoch(rec.as_row(model=spec.name, recipe_digest=digest))
        print(f"    epoch {rec.epoch:3d}/{recipe.epochs}  loss {rec.train_loss:.3f}  "
              f"top1 {rec.top1:6.2f}  top5 {rec.top5:6.2f}  "
              f"{rec.epoch_seconds:.0f}s  {rec.images_per_second:.0f} img/s")

    trainer = Trainer(spec, recipe, run, train_cache, val_cache,
                      resume_path=resume_path, on_epoch=record_epoch)
    result = trainer.fit()

    # [AUDIT T9] Provenance records the kwargs the registry actually built with, including
    # forced ones such as aux_logits=False, rather than only those the spec requested.
    # Before:
    #   kwargs=dict(spec.kwargs), num_classes=recipe.num_classes,
    meta = CheckpointMeta(
        name=spec.name, source=result.resolved.source,
        provider_version=result.resolved.provider_version,
        kwargs=dict(result.resolved.kwargs), num_classes=recipe.num_classes,
        classes=list(train_cache.classes), recipe_digest=digest,
        task=exp.task, dataset=exp.dataset, epochs_completed=result.epochs_completed,
        metrics={"top1": result.final.top1, "top5": result.final.top5,
                 "val_loss": result.final.val_loss},
        extra={"best_top1": result.best_top1, "best_epoch": result.best_epoch,
               "also_available_in": list(result.resolved.also_available_in)},
    )
    save_checkpoint(final_path, result.resolved.module, meta, recipe)

    # [AUDIT P3] The resume file was deleted before the completed row was recorded (which
    # happened back in main). A failure or kill in between left a .pt with no completed row,
    # so the model retrained from scratch with its resume file gone. The row is now recorded
    # first and the resume file removed last.
    # Before:
    #   resume_path.unlink(missing_ok=True)
    #   row = summary_row(spec, result, recipe, run, exp.dataset, final_path, paths)
    #   store.write_sidecar(final_path, {**meta.to_dict(), "recipe": recipe.to_dict()})
    #   return row
    row = summary_row(spec, result, recipe, run, exp.dataset, final_path, paths)
    store.write_sidecar(final_path, {**meta.to_dict(), "recipe": recipe.to_dict()})
    store.record(row)
    # The resume file is scratch; removing it on success keeps the checkpoint directory free of
    # files that look like deliverables but are not.
    resume_path.unlink(missing_ok=True)
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

    # [AUDIT P4] Overrides were assigned field by field, bypassing RunConfig validation:
    # --limit-batches 0 became "no limit" (a full epoch), --micro-batch-size 0 divided by zero,
    # and a batch size not divisible by the micro-batch failed separately inside every model's
    # Trainer. Overrides now go through replace(), which validates, and the accumulation split
    # is checked once here, before any training starts.
    # Before:
    #   if args.micro_batch_size is not None:
    #       exp.run.micro_batch_size = args.micro_batch_size
    #   if args.device:
    #       exp.run.device = args.device
    #   ... (same pattern for limit_batches, debug, resume, compile)
    run_overrides: dict[str, Any] = {}
    if args.micro_batch_size is not None:
        run_overrides["micro_batch_size"] = args.micro_batch_size
    if args.device:
        run_overrides["device"] = args.device
    if args.limit_batches is not None:
        run_overrides["limit_batches"] = args.limit_batches
    if args.debug:
        run_overrides["debug"] = True
    if args.no_resume:
        run_overrides["resume"] = False
    if args.compile:
        run_overrides["compile"] = True
    if run_overrides:
        exp.run = dataclasses.replace(exp.run, **run_overrides)
    exp.run.accumulation_steps(exp.recipe)

    if not exp.models:
        raise ConfigError("no models specified; use --models or a config file")
    return exp


def parse_split_dirs(items: Sequence[str] | None) -> dict[str, list[str]]:
    """Apply --split-dir overrides to the default split layout.

    Args:
        items: Values of the form "SPLIT=DIR1,DIR2".

    Returns:
        Mapping of split name to its sub-directories.

    Raises:
        ConfigError: If an item lacks "=", names no directories, or contains an empty one.
    """
    # [AUDIT P5] An item without "=" (e.g. "train") silently mapped the split to the dataset
    # root itself.
    # Before (inline in main):
    #   for item in args.split_dir or []:
    #       split, _, dirs = item.partition("=")
    #       split_dirs[split] = dirs.split(",")
    split_dirs = dict(DEFAULT_SPLIT_DIRS)
    for item in items or []:
        split, sep, dirs = item.partition("=")
        parts = dirs.split(",")
        if not sep or not split or not all(parts):
            raise ConfigError(f"--split-dir expects SPLIT=DIR1,DIR2, got {item!r}")
        split_dirs[split] = parts
    return split_dirs


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
    ap.add_argument("--label-from", choices=LABEL_FROM, default="parent",
                    help="read labels from the directory containing each image (parent, the "
                         "Kaggle ImageNet-100 layout) or the first directory below the split "
                         "root (top, e.g. Tiny-ImageNet's <class>/images/)")
    args = ap.parse_args(argv)

    # [AUDIT P5] --split-dir parsing moved into parse_split_dirs, validated with the rest of
    # the configuration.
    # Before:
    #   try:
    #       exp = build_experiment(args)
    #   except ConfigError as exc:
    #       ...
    #   split_dirs = dict(DEFAULT_SPLIT_DIRS)
    #   for item in args.split_dir or []: ...
    try:
        exp = build_experiment(args)
        split_dirs = parse_split_dirs(args.split_dir)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    recipe, run = exp.recipe, exp.run
    print(f"recipe  {recipe.summary()}")
    print(f"dataset {exp.dataset}   task {exp.task}   debug={run.debug}")

    # [AUDIT P5] PathError (no project root, unsafe dataset name), a CacheError from opening
    # the caches, and RegistryError/ResultsError used to escape as raw tracebacks. They now
    # end the run with a message and exit status 2, like the other setup errors.
    # Before:
    #   paths = get_paths()
    #   try:
    #       ensure_caches(paths, exp.dataset, recipe, split_dirs, rebuild=args.rebuild_cache)
    #   except CacheError as exc:
    #       ...
    #   train_cache = ImageCache.open(cache_dir, "train", recipe.train_res)
    #   val_cache = ImageCache.open(cache_dir, "val", recipe.eval_res)
    #   ...
    #   specs = [ModelSpec.coerce(m) for m in exp.models]
    try:
        paths = get_paths()
        ensure_caches(paths, exp.dataset, recipe, split_dirs, rebuild=args.rebuild_cache,
                      label_from=args.label_from, workers=run.workers)
        if args.cache_only:
            print("caches ready; --cache-only requested, stopping here")
            return 0
        cache_dir = paths.cache_dir(exp.dataset)
        train_cache = ImageCache.open(cache_dir, "train", recipe.train_cache_resolution)
        val_cache = ImageCache.open(cache_dir, "val", recipe.eval_res)
        specs = [ModelSpec.coerce(m) for m in exp.models]
        store = make_store(paths, run.debug)
    except (CacheError, PathError, RegistryError, ResultsError) as exc:
        print(f"setup error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"  {train_cache.describe()}")
    print(f"  {val_cache.describe()}")

    # [AUDIT P6] A model listed twice was trained twice. Duplicates by name are now dropped,
    # keeping the first entry, with a warning.
    # Before: (no de-duplication)
    unique: dict[str, ModelSpec] = {}
    for s in specs:
        if s.name in unique:
            print(f"WARNING: {s.name} is listed more than once; using the first entry",
                  file=sys.stderr)
        else:
            unique[s.name] = s
    specs = list(unique.values())
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
    names = [s.name for s in specs]
    force = names if args.force_all else args.force

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
        # [AUDIT P3] train_one records its completed row itself, before deleting the resume
        # file; only failures are recorded here.
        # Before:
        #   try:
        #       row = train_one(...)
        #       ...
        #   except Exception as exc:
        #       ...
        #   store.record(row)
        try:
            row = train_one(spec, exp, paths, store, train_cache, val_cache)
            print(f"  done: top1 {row['top1']:.2f}  top5 {row['top5']:.2f}  "
                  f"{row['train_minutes']:.1f} min")
        except Exception as exc:  # one model must not end the sweep
            message = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {message}", file=sys.stderr)
            traceback.print_exc()
            failures.append(name)
            store.record({"model": name, "status": STATUS_FAILED,
                          "recipe_digest": recipe.digest(), "error": message[:400]})

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