#!/usr/bin/env python3
"""
FILENAME:       src/edgexbench/eval/baseline.py

DESCRIPTION:    FP32 reference measurements for trained checkpoints on the development GPU.
                Establishes the accuracy every quantised result is compared against, measured
                through the same procedure the hardware backends will use. The accuracy written
                during training is not a valid reference: it was produced under autocast inside
                the training loop, so part of any later gap would be the harness rather than the
                quantisation.
                Latency here is a reference point, not a platform result. A development GPU is
                not an edge device, so these timings belong beside the accelerator figures only
                as an unconstrained upper bound.
AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-30
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from edgexbench.core.artifacts import load_checkpoint, read_meta, rebuild_model
from edgexbench.core.data.cache import ImageCache
from edgexbench.core.data.loader import Normalisation, centre_view, normalise
from edgexbench.core.data.sampling import SubsetSpec, random_subset
from edgexbench.core.paths import Paths, ensure_dir, get_paths
from edgexbench.core.results import (STATUS_COMPLETED, STATUS_FAILED, ResultsLayout,
                                     ResultsStore)

TASK = "classification"

# Reported alongside every latency figure. Latency is measured at batch 1 because that is the
# deployment case for a real-time pipeline; throughput at a larger batch answers a different
# question and is reported separately.
LATENCY_BATCH = 1


class BaselineError(RuntimeError):
    """Raised when a model cannot be evaluated against the available caches."""


# -----------------------------------------------------------------------------------------
# Complexity
# -----------------------------------------------------------------------------------------
def complexity(model: nn.Module, resolution: int) -> tuple[float, float]:
    """Measure the theoretical cost of one forward pass.

    NOTE: the operation counter tallies convolutions, matrix multiplications and attention
    only. Normalisation, activations, pooling, concatenation and channel shuffle are excluded.
    This matches the convention used in the literature, and is precisely why multiply-accumulate
    counts predict latency poorly for shuffle- and depthwise-heavy architectures.

    Args:
        model: Model to measure. Copied, so the original is untouched.
        resolution: Input side length.

    Returns:
        Parameter count in millions, and multiply-accumulate count in billions.
    """
    from torch.utils.flop_counter import FlopCounterMode

    probe = copy.deepcopy(model).eval().cpu()
    params = sum(p.numel() for p in probe.parameters())
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        # Input values are irrelevant: the count depends only on tensor shapes.
        probe(torch.zeros(1, 3, resolution, resolution))
    # Halved because one multiply-accumulate is two floating-point operations.
    return params / 1e6, counter.get_total_flops() / 2 / 1e9


# -----------------------------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------------------------
def topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> tuple[int, int]:
    """Count top-1 and top-k correct predictions in one batch.

    Args:
        logits: Model outputs, shaped (batch, classes).
        labels: Ground-truth class indices.
        k: Size of the top-k set, clamped to the class count.

    Returns:
        The top-1 and top-k hit counts.
    """
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    hit = top == labels[:, None]
    return int(hit[:, 0].sum()), int(hit.any(dim=1).sum())


@torch.inference_mode()
def evaluate(model: nn.Module, cache: ImageCache, device: torch.device, batch_size: int,
             norm: Normalisation, indices: np.ndarray | None = None,
             channels_last: bool = True) -> dict[str, Any]:
    """Score a model over the validation cache in true FP32.

    Autocast is deliberately not used, unlike during training: this is the reference every
    reduced-precision result is subtracted from, so it must be the unreduced one.

    Args:
        model: Model in evaluation mode.
        cache: Validation image cache.
        device: Device to run on.
        batch_size: Items per forward pass.
        norm: Channel normalisation, matching the training recipe.
        indices: Subset of cache indices to score. None scores everything.
        channels_last: Use the channels-last memory format.

    Returns:
        Mapping with top-1, top-5, mean loss, the number of images scored, and the elapsed time.

    Raises:
        BaselineError: If the number of images scored differs from the number requested.
    """
    model.eval()
    order = np.arange(len(cache)) if indices is None else np.asarray(indices)
    expected = len(order)

    hits1 = hits5 = seen = 0
    loss_sum = 0.0
    started = time.perf_counter()

    for start in range(0, expected, batch_size):
        batch_idx = order[start:min(start + batch_size, expected)]
        images, labels = cache.take(batch_idx)
        x = torch.from_numpy(images).to(device, non_blocking=True)
        y = torch.from_numpy(labels).to(device, non_blocking=True)

        # No augmentation: validation images were resized and cropped when the cache was built.
        x = normalise(centre_view(x), norm)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        logits = model(x).float()
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        a, b = topk_hits(logits, y)
        hits1 += a
        hits5 += b
        seen += len(y)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Counted rather than assumed: dividing by an assumed total would silently distort the
    # accuracy if the loop ever covered a different number of images than intended.
    if seen != expected:
        raise BaselineError(f"scored {seen} images, expected {expected}")

    return {"top1": round(100.0 * hits1 / seen, 3),
            "top5": round(100.0 * hits5 / seen, 3),
            "val_loss": round(loss_sum / seen, 5),
            "n_eval": seen,
            "eval_seconds": round(time.perf_counter() - started, 1)}


# -----------------------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------------------
def latency_stats(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Summarise a set of per-iteration timings.

    Percentiles are reported rather than the mean alone because a real-time pipeline must
    budget for the tail, which the mean conceals.

    Args:
        samples_ms: Per-iteration durations in milliseconds.

    Returns:
        Mapping of statistic name to value.
    """
    a = np.asarray(samples_ms, dtype=np.float64)
    return {"lat_mean_ms": round(float(a.mean()), 4),
            "lat_p50_ms": round(float(np.percentile(a, 50)), 4),
            "lat_p95_ms": round(float(np.percentile(a, 95)), 4),
            "lat_p99_ms": round(float(np.percentile(a, 99)), 4),
            "lat_min_ms": round(float(a.min()), 4),
            "lat_std_ms": round(float(a.std()), 4),
            "lat_iters": int(a.size)}


@torch.inference_mode()
def measure_latency(model: nn.Module, resolution: int, device: torch.device,
                    warmup: int, iters: int, norm: Normalisation,
                    channels_last: bool = True) -> dict[str, Any]:
    """Measure single-sample inference latency.

    Each iteration is timed individually so that percentiles are meaningful. On CUDA the timed
    region is bracketed by synchronisation: kernel launches are asynchronous, so without it the
    clock would measure how quickly work was queued rather than how long it took to run.

    Args:
        model: Model in evaluation mode.
        resolution: Input side length.
        device: Device to run on.
        warmup: Iterations discarded before timing, absorbing autotuning and allocation.
        iters: Timed iterations.
        norm: Channel normalisation, applied so the measured graph matches the scored one.
        channels_last: Use the channels-last memory format.

    Returns:
        Latency statistics.
    """
    model.eval()
    # Fixed seed: every model and every repeat sees an identical input, removing input content
    # as a source of variation between measurements.
    generator = torch.Generator().manual_seed(0)
    raw = torch.randint(0, 256, (LATENCY_BATCH, resolution, resolution, 3),
                        dtype=torch.uint8, generator=generator).to(device)
    x = normalise(centre_view(raw), norm)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return latency_stats(samples)


@torch.inference_mode()
def measure_throughput(model: nn.Module, resolution: int, device: torch.device,
                       batch_size: int, iters: int, norm: Normalisation,
                       channels_last: bool = True) -> dict[str, Any]:
    """Measure sustained images per second at a larger batch.

    A different question from single-sample latency: this reports how well an architecture
    fills the device. Memory-bound designs, which includes most depthwise-heavy mobile models,
    score far worse here than their operation counts suggest.

    The batch is halved on an out-of-memory failure rather than abandoning the model, and the
    batch actually used is reported so two figures are never silently incomparable.

    Args:
        model: Model in evaluation mode.
        resolution: Input side length.
        device: Device to run on.
        batch_size: Starting batch size.
        iters: Timed iterations.
        norm: Channel normalisation.
        channels_last: Use the channels-last memory format.

    Returns:
        Mapping with the batch size used and the measured throughput.
    """
    model.eval()
    while batch_size >= 1:
        try:
            raw = torch.randint(0, 256, (batch_size, resolution, resolution, 3),
                                dtype=torch.uint8, device=device)
            x = normalise(centre_view(raw), norm)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            for _ in range(3):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(iters):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            return {"throughput_batch": batch_size,
                    "throughput_img_s": round(batch_size * iters / elapsed, 1)}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"      throughput out of memory, retrying at batch {batch_size}")
    return {"throughput_batch": None, "throughput_img_s": None}


# -----------------------------------------------------------------------------------------
# Per-model driver
# -----------------------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    """Measurement settings for one baseline run.

    Attributes:
        eval_batch: Batch size for the accuracy pass.
        warmup: Discarded latency iterations.
        iters: Timed latency iterations.
        throughput_batch: Starting batch for the throughput measurement.
        throughput_iters: Timed throughput iterations.
        eval_n: Number of validation images to score; 0 scores the full set.
        subset_seed: Seed for the validation subset, when one is used.
        channels_last: Use the channels-last memory format.
        skip_latency: Measure accuracy only.
    """

    eval_batch: int = 128
    warmup: int = 50
    iters: int = 500
    throughput_batch: int = 64
    throughput_iters: int = 30
    eval_n: int = 0
    subset_seed: int = 0
    channels_last: bool = True
    skip_latency: bool = False


def open_val_cache(paths: Paths, dataset: str, resolution: int,
                   opened: dict[int, ImageCache]) -> ImageCache:
    """Open the validation cache for one resolution, reusing an already-open handle.

    Models in a sweep may have been trained at different evaluation resolutions, and each must
    be scored against images at its own size. Scoring a model against the wrong resolution
    produces a plausible but incorrect accuracy, which is the hardest kind of error to notice.

    Args:
        paths: Resolved project layout.
        dataset: Dataset identifier.
        resolution: Required image resolution.
        opened: Cache of already-open handles, keyed by resolution.

    Returns:
        The validation cache at the requested resolution.
    """
    if resolution not in opened:
        opened[resolution] = ImageCache.open(paths.cache_dir(dataset), "val", resolution)
    return opened[resolution]


def baseline_one(checkpoint: Path, paths: Paths, dataset: str, device: torch.device,
                 cfg: BaselineConfig, norm: Normalisation,
                 opened: dict[int, ImageCache]) -> dict[str, Any]:
    """Measure one trained checkpoint.

    Args:
        checkpoint: Path to the checkpoint.
        paths: Resolved project layout.
        dataset: Dataset identifier.
        device: Device to run on.
        cfg: Measurement settings.
        norm: Channel normalisation matching the training recipe.
        opened: Open validation caches, keyed by resolution.

    Returns:
        The results row for this model.

    Raises:
        BaselineError: If the checkpoint records no evaluation resolution.
    """
    payload = load_checkpoint(checkpoint, map_location="cpu")
    meta = read_meta(payload, fallback_name=checkpoint.stem)
    recipe = payload.get("recipe") or {}
    resolution = int(recipe.get("eval_res") or 0)
    if not resolution:
        raise BaselineError(
            f"{checkpoint.name} records no eval_res; it cannot be scored without knowing the "
            f"resolution it was trained for")

    model, _ = rebuild_model(payload, fallback_name=checkpoint.stem)
    params_m, gmacs = complexity(model, resolution)
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    cache = open_val_cache(paths, dataset, resolution, opened)

    # The cache is written in class-sorted order, so a prefix of it is the first few classes
    # rather than a sample of the dataset. Any subset must therefore be drawn at random.
    indices = random_subset(len(cache), cfg.eval_n, seed=cfg.subset_seed)
    subset = (SubsetSpec.full(len(cache)) if indices is None
              else SubsetSpec.from_indices("random", indices, len(cache), cfg.subset_seed))

    row: dict[str, Any] = {
        "model": checkpoint.stem,
        "status": STATUS_COMPLETED,
        "artifact": "pytorch-fp32",
        "precision": "fp32",
        "recipe_digest": meta.recipe_digest,
        "source": meta.source,
        "provider_version": meta.provider_version,
        "dataset": dataset,
        "res": resolution,
        "num_classes": meta.num_classes,
        "params_M": round(params_m, 3),
        "GMACs": round(gmacs, 4),
        "checkpoint_mb": round(checkpoint.stat().st_size / 2 ** 20, 2),
        # The training figure was measured under autocast inside the training loop. Kept for
        # comparison only; it is not a substitute for the FP32 measurement below.
        "train_top1": meta.metrics.get("top1"),
        "error": "",
    }
    row.update(subset.as_dict())
    row.update(evaluate(model, cache, device, cfg.eval_batch, norm, indices, cfg.channels_last))
    row["top1_delta_vs_training"] = (round(row["top1"] - row["train_top1"], 3)
                                     if row["train_top1"] is not None else None)

    if not cfg.skip_latency:
        # Memory statistics are reset after the accuracy pass so the recorded peak reflects
        # single-sample inference, which is the deployment-relevant figure; the accuracy pass
        # uses a large batch and would otherwise dominate it.
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        row.update(measure_latency(model, resolution, device, cfg.warmup, cfg.iters, norm,
                                   cfg.channels_last))
        if device.type == "cuda":
            row["peak_memory_bs1_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        row.update(measure_throughput(model, resolution, device, cfg.throughput_batch,
                                      cfg.throughput_iters, norm, cfg.channels_last))

    row["device"] = (torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu")
    row["torch_version"] = torch.__version__
    row["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


# -----------------------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    """Measure every trained checkpoint and write the baseline results table.

    Args:
        argv: Command-line arguments. None reads sys.argv.

    Returns:
        Process exit status: 0 when every model was measured, 1 otherwise.
    """
    ap = argparse.ArgumentParser(
        description="FP32 reference accuracy and timing for trained checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--dataset", default="imageNet-100")
    ap.add_argument("--only", nargs="*", help="restrict to these model names")
    ap.add_argument("--force", nargs="*", default=[],
                    help="re-measure these models even if already recorded")
    ap.add_argument("--force-all", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--eval-n", type=int, default=0,
                    help="validation images to score; 0 scores the full set. Any subset is "
                         "drawn at random, never as a prefix, because the cache is class-sorted")
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--throughput-batch", type=int, default=64)
    ap.add_argument("--throughput-iters", type=int, default=30)
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="read debug checkpoints and write to an isolated results directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # External architectures must be registered before any checkpoint referring to one can be
    # rebuilt; a missing registration would surface as an unresolvable architecture name.
    try:
        from edgexbench.models.external import register_all
        register_all()
    except ImportError:
        pass

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device: {torch.cuda.get_device_name(0)}")
        print("  NOTE: lock the clocks before a run whose timings will be reported, since a "
              "consumer card throttles under sustained load")
    else:
        print("device: cpu   (timings are not comparable to a GPU run)")

    paths = get_paths()
    suffix = "_debug" if args.debug else ""
    checkpoint_dir = paths.checkpoint_dir("pytorch", args.task + suffix)
    if not checkpoint_dir.is_dir():
        print(f"no checkpoint directory: {checkpoint_dir}", file=sys.stderr)
        return 2

    checkpoints = sorted(p for p in checkpoint_dir.glob("*.pt")
                         if not p.name.endswith(".resume.pt"))
    if args.only:
        wanted = set(args.only)
        checkpoints = [p for p in checkpoints if p.stem in wanted]
    if not checkpoints:
        print("no checkpoints to measure", file=sys.stderr)
        return 2

    results_dir = ensure_dir(paths.results_dir("benchmark", args.task + suffix))
    store = ResultsStore(ResultsLayout(results_dir=results_dir,
                                       artifact_dir=checkpoint_dir,
                                       summary_name="baseline_fp32.csv",
                                       epochs_name="baseline_fp32_epochs.csv"))

    names = [p.stem for p in checkpoints]
    recorded = set()
    existing = store.summary()
    if not existing.empty and "model" in existing.columns:
        recorded = {str(m) for m in existing["model"]}
    force = set(names if args.force_all else args.force)
    todo = [p for p in checkpoints if p.stem in force or p.stem not in recorded]

    print(f"\ncheckpoints : {len(checkpoints)}")
    print(f"already done: {len(checkpoints) - len(todo)}")
    print(f"to measure  : {len(todo)} {[p.stem for p in todo]}")
    if args.dry_run:
        return 0

    cfg = BaselineConfig(eval_batch=args.eval_batch, warmup=args.warmup, iters=args.iters,
                         throughput_batch=args.throughput_batch,
                         throughput_iters=args.throughput_iters, eval_n=args.eval_n,
                         subset_seed=args.subset_seed, skip_latency=args.skip_latency)
    norm = Normalisation()
    opened: dict[int, ImageCache] = {}
    failures: list[str] = []

    for i, checkpoint in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}] {checkpoint.stem}")
        try:
            row = baseline_one(checkpoint, paths, args.dataset, device, cfg, norm, opened)
            print(f"      top1 {row['top1']:6.2f}  top5 {row['top5']:6.2f}  "
                  f"({row['n_eval']} imgs, {row['res']}px)")
            if row.get("top1_delta_vs_training") is not None:
                print(f"      vs training-time figure: "
                      f"{row['top1_delta_vs_training']:+.2f} pp")
            if not cfg.skip_latency:
                print(f"      p50 {row['lat_p50_ms']:.2f} ms   p99 {row['lat_p99_ms']:.2f} ms   "
                      f"{row['throughput_img_s']} img/s   "
                      f"{row['params_M']:.2f} M params   {row['GMACs']:.3f} GMACs")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"      FAILED: {message}", file=sys.stderr)
            traceback.print_exc()
            failures.append(checkpoint.stem)
            row = {"model": checkpoint.stem, "status": STATUS_FAILED,
                   "artifact": "pytorch-fp32", "error": message[:400]}
        store.record(row)

    digests = store.digests()
    if len(digests) > 1:
        print(f"\nWARNING: these results span {len(digests)} recipe digests {digests}. "
              f"Rows trained under different recipes are not directly comparable.")

    summary = store.summary()
    good = summary[summary["status"] == STATUS_COMPLETED] if "status" in summary else summary
    if not good.empty:
        columns = [c for c in ("model", "top1", "top5", "params_M", "GMACs",
                               "lat_p50_ms", "throughput_img_s") if c in good.columns]
        print("\nRanked by accuracy:")
        print(good[columns].sort_values("top1", ascending=False).to_string(index=False))
        if "lat_p50_ms" in columns:
            print("\nRanked by latency:")
            print(good[columns].sort_values("lat_p50_ms").to_string(index=False))

    print(f"\nwrote {store.layout.summary_path}")
    if failures:
        print(f"failed: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())