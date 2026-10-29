"""
FILENAME:       src/edgebench/train/trainer.py

DESCRIPTION:    Training loop for one model under one recipe. A single implementation is used
                for every architecture, which is what makes the resulting accuracies a
                comparison of architectures rather than of training procedures. Supports
                gradient accumulation so that models which do not fit in memory are still
                trained at the recipe's effective batch size, and resumes from a per-epoch
                checkpoint pinned to the recipe digest.
AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from edgebench.core.artifacts import CheckpointMeta, load_checkpoint, save_checkpoint
from edgebench.core.config import RecipeConfig, RunConfig
from edgebench.core.data.cache import ImageCache
from edgebench.core.data.loader import (CacheLoader, Normalisation, centre_view, normalise,
                                        random_resized_crop_flip)
from edgebench.core.data.sampling import random_subset
from edgebench.core.registry import ModelRegistry, ModelSpec, ResolvedModel, get_registry

# Micro-batches between loss finiteness checks when log_every is 0. Each check is one
# device-to-host sync, so this is frequent enough to stop a diverged run early and rare enough
# to cost nothing measurable.
FINITE_CHECK_EVERY: int = 100


class TrainingError(RuntimeError):
    """Raised when training cannot proceed or produces a non-finite loss."""


@dataclass
class EpochRecord:
    """Metrics from one completed epoch.

    Attributes:
        epoch: One-based epoch number.
        train_loss: Mean training loss over the epoch.
        val_loss: Mean validation loss.
        top1: Validation top-1 accuracy, as a percentage.
        top5: Validation top-5 accuracy, as a percentage.
        lr: Learning rate at the end of the epoch.
        epoch_seconds: Wall-clock duration of the training pass.
        images_per_second: Training throughput.
    """

    epoch: int
    train_loss: float
    val_loss: float
    top1: float
    top5: float
    lr: float
    epoch_seconds: float
    images_per_second: float

    def as_row(self, **extra: Any) -> dict[str, Any]:
        """Flatten for the per-epoch results table.

        Args:
            **extra: Additional columns, typically the model identifier.

        Returns:
            Mapping of column name to value.
        """
        return {"epoch": self.epoch, "train_loss": self.train_loss, "val_loss": self.val_loss,
                "top1": self.top1, "top5": self.top5, "lr": self.lr,
                "epoch_seconds": self.epoch_seconds,
                "images_per_second": self.images_per_second, **extra}


@dataclass
class TrainingResult:
    """Outcome of a completed training run.

    Attributes:
        model: Architecture identifier.
        resolved: The implementation that was trained.
        epochs_completed: Epochs trained in total, including any resumed from.
        final: Metrics from the last epoch. These are the reported figures.
        best_top1: Highest validation top-1 observed.
        best_epoch: Epoch at which best_top1 occurred.
        train_minutes: Wall-clock training time in this session.
        seconds_per_epoch: Mean epoch duration in this session.
        peak_memory_mb: Peak device memory allocated.
        history: Every epoch record from this session.
    """

    model: str
    resolved: ResolvedModel
    epochs_completed: int
    final: EpochRecord
    best_top1: float
    best_epoch: int
    train_minutes: float
    seconds_per_epoch: float
    peak_memory_mb: float | None
    history: list[EpochRecord] = field(default_factory=list)


def resolve_device(spec: str) -> torch.device:
    """Select the compute device.

    Args:
        spec: Device string, or "auto" to prefer CUDA when present.

    Returns:
        The selected device.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _get_rng_state() -> dict[str, Any]:
    """Capture every random stream that training draws from.

    Returns:
        Python, NumPy, torch CPU and, when present, CUDA generator states.
    """
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any] | None) -> None:
    """Restore random streams captured by _get_rng_state.

    Args:
        state: Captured states, or None to leave the current streams untouched.

    Returns:
        None.
    """
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda])


def seed_everything(seed: int) -> None:
    """Seed the random sources that affect weight initialisation and data order.

    NOTE: this does not make a run bit-reproducible. cuDNN benchmarking selects algorithms by
    timing, and some kernels accumulate non-deterministically, so two runs with one seed can
    differ slightly. Estimate that noise across a few seeds before treating a small accuracy
    gap between architectures as real.

    Args:
        seed: Seed value.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: nn.Module, recipe: RecipeConfig) -> torch.optim.Optimizer:
    """Construct the optimiser with weight decay applied selectively.

    Decay is applied to convolution and linear weights only. Normalisation gains and biases are
    one-dimensional and decaying them measurably harms accuracy; excluding them is standard in
    modern ImageNet recipes.

    Args:
        model: Model whose parameters to optimise.
        recipe: Recipe supplying the optimiser settings.

    Returns:
        The configured optimiser.

    Raises:
        TrainingError: If the recipe names an unsupported optimiser.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    groups = [{"params": decay, "weight_decay": recipe.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]

    if recipe.optimizer == "adamw":
        # [AUDIT T7] Fused kernels were chosen by CUDA availability, so --device cpu on a GPU
        # machine requested fused AdamW over CPU parameters, which torch < 2.4 rejects.
        # Before:
        #   fused = torch.cuda.is_available()
        fused = next(model.parameters()).is_cuda
        return torch.optim.AdamW(groups, lr=recipe.lr, fused=fused)
    if recipe.optimizer == "sgd":
        # [AUDIT T6] Nesterov requires non-zero momentum, which the recipe allows to be 0.
        # Before:
        #   return torch.optim.SGD(groups, lr=recipe.lr, momentum=recipe.momentum, nesterov=True)
        return torch.optim.SGD(groups, lr=recipe.lr, momentum=recipe.momentum,
                               nesterov=recipe.momentum > 0)
    raise TrainingError(f"unsupported optimizer {recipe.optimizer!r}")


def lr_at_step(step: int, steps_per_epoch: int, recipe: RecipeConfig) -> float:
    """Compute the learning rate for one optimiser step.

    Evaluated per step rather than per epoch so the warmup ramp and the decay curve are smooth
    rather than a staircase.

    Args:
        step: Zero-based global optimiser step.
        steps_per_epoch: Optimiser steps in one epoch.
        recipe: Recipe supplying the schedule.

    Returns:
        The learning rate to apply.
    """
    warmup_steps = recipe.warmup_epochs * steps_per_epoch
    total_steps = recipe.epochs * steps_per_epoch

    if warmup_steps and step < warmup_steps:
        return recipe.lr * (step + 1) / warmup_steps

    if recipe.scheduler == "constant":
        return recipe.lr

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)

    if recipe.scheduler == "cosine":
        return recipe.min_lr + 0.5 * (recipe.lr - recipe.min_lr) * (1 + math.cos(math.pi * progress))
    if recipe.scheduler == "step":
        # Three decades of decay across the run, a conventional step schedule.
        return max(recipe.min_lr, recipe.lr * (0.1 ** int(progress * 3)))
    return recipe.lr


def topk_hits(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> tuple[int, int]:
    """Count top-1 and top-k correct predictions in one batch.

    Args:
        logits: Model outputs, shaped (batch, classes).
        labels: Ground-truth class indices.
        k: Size of the top-k set, clamped to the number of classes.

    Returns:
        The top-1 and top-k hit counts.
    """
    top1, topk = topk_hit_tensors(logits, labels, k)
    return int(top1), int(topk)


def topk_hit_tensors(logits: torch.Tensor, labels: torch.Tensor,
                     k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Count top-1 and top-k correct predictions without leaving the device.

    Args:
        logits: Model outputs, shaped (batch, classes).
        labels: Ground-truth class indices.
        k: Size of the top-k set, clamped to the number of classes.

    Returns:
        The top-1 and top-k hit counts as zero-dimensional tensors on the logits' device.
    """
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    hit = top == labels[:, None]
    return hit[:, 0].sum(), hit.any(dim=1).sum()


def _as_logits(output: Any) -> torch.Tensor:
    """Check that a forward pass produced a single logits tensor.

    Args:
        output: Value returned by the model.

    Returns:
        The output, unchanged.

    Raises:
        TrainingError: If the model returned a tuple or other non-tensor value.
    """
    # [AUDIT T9] A tuple output (auxiliary heads) used to crash at logits.float() with an
    # AttributeError. The registry now builds such models single-headed; anything that still
    # returns several outputs fails loudly here rather than having a head silently dropped.
    # Before: logits = forward(x); logits = logits.float()
    if not torch.is_tensor(output):
        raise TrainingError(
            f"model returned {type(output).__name__} (a tuple of outputs?) instead of a logits "
            f"tensor; build it without auxiliary heads, e.g. kwargs aux_logits=False")
    return output


class Trainer:
    """Trains one model under one recipe."""

    def __init__(self,
                 spec: ModelSpec,
                 recipe: RecipeConfig,
                 run: RunConfig,
                 train_cache: ImageCache,
                 val_cache: ImageCache,
                 *,
                 registry: ModelRegistry | None = None,
                 normalisation: Normalisation | None = None,
                 resume_path: Path | None = None,
                 on_epoch: Callable[[EpochRecord], None] | None = None) -> None:
        """Initialise the trainer.

        Args:
            spec: Architecture to train, or any form accepted by ModelSpec.coerce.
            recipe: Training recipe.
            run: Execution settings.
            train_cache: Cache supplying training images, at recipe.train_cache_resolution.
            val_cache: Cache supplying validation images, at recipe.eval_res.
            registry: Registry to resolve the architecture through.
            normalisation: Channel statistics. Defaults to ImageNet-1k.
            resume_path: Checkpoint to resume from and write to after each epoch.
            on_epoch: Callback invoked with each completed epoch record, used to stream rows
                into the results table so a crash does not lose the learning curve.

        Raises:
            TrainingError: If a cache resolution does not match the recipe, the class counts
                or class lists disagree, or mixing is enabled with an odd micro-batch.
        """
        # [AUDIT P1] The training cache is now stored larger than train_res, so the random
        # crop samples from more detail than it outputs. Its side is a recipe parameter, so the
        # cache must match it exactly.
        # Before:
        #   if train_cache.resolution != recipe.train_res:
        #       raise TrainingError(
        #           f"train cache is {train_cache.resolution}px, recipe wants {recipe.train_res}px")
        if train_cache.resolution != recipe.train_cache_resolution:
            raise TrainingError(
                f"train cache is {train_cache.resolution}px, recipe wants "
                f"{recipe.train_cache_resolution}px (train_cache_res)")
        if val_cache.resolution != recipe.eval_res:
            raise TrainingError(
                f"val cache is {val_cache.resolution}px, recipe wants {recipe.eval_res}px")
        if len(train_cache.classes) != recipe.num_classes:
            raise TrainingError(
                f"train cache has {len(train_cache.classes)} classes, "
                f"recipe wants {recipe.num_classes}")
        # [AUDIT T8] Only the class count was compared, so a val cache built against a
        # different class ordering scored with shifted labels and produced a plausible but
        # wrong accuracy.
        # Before: (no check)
        if list(train_cache.classes) != list(val_cache.classes):
            raise TrainingError(
                "train and val caches disagree on the class list or its order; rebuild the val "
                "cache against the train cache's classes")

        # [AUDIT T11] fit() read self.spec.name, so a plain string spec (accepted by
        # registry.resolve) crashed only after training had finished.
        # Before:
        #   self.spec = spec
        self.spec = ModelSpec.coerce(spec)
        self.recipe = recipe
        self.run = run
        self.train_cache = train_cache
        self.val_cache = val_cache
        self.registry = registry or get_registry()
        self.norm = normalisation or Normalisation()
        self.resume_path = Path(resume_path) if resume_path else None
        self.on_epoch = on_epoch

        self.device = resolve_device(run.device)
        self.amp_dtype = getattr(torch, recipe.amp_dtype)
        self.accum = run.accumulation_steps(recipe)
        self.micro_bs = run.micro_batch_size or recipe.batch_size
        # [AUDIT T10] timm's Mixup asserts on an odd batch, mid-training.
        # Before: (no check)
        if recipe.mix_enabled and self.micro_bs % 2:
            raise TrainingError(
                f"mixup/cutmix needs an even micro-batch size, got {self.micro_bs}")

    #  mixup

    def _make_mixer(self):
        """Construct the mixup and CutMix transform, if the recipe enables it.

        timm's implementation is used rather than a local one because it is the reference
        these recipes were tuned against, and its soft-target handling is already matched to
        label smoothing.

        Returns:
            The configured Mixup object, or None when mixing is disabled.

        Raises:
            TrainingError: If the recipe enables mixing but timm is not installed.
        """
        if not self.recipe.mix_enabled:
            return None
        try:
            from timm.data import Mixup
        except ImportError as exc:
            raise TrainingError(
                "recipe enables mixup/cutmix but timm is not installed: pip install timm") from exc
        return Mixup(mixup_alpha=self.recipe.mixup_alpha,
                     cutmix_alpha=self.recipe.cutmix_alpha,
                     prob=self.recipe.mix_prob, switch_prob=0.5, mode="batch",
                     label_smoothing=self.recipe.label_smoothing,
                     num_classes=self.recipe.num_classes)

    # ---- passes -------------------------------------------------------------------------

    def _train_epoch(self, model: nn.Module, forward: nn.Module,
                     optimizer: torch.optim.Optimizer, loader: CacheLoader,
                     mixer, epoch: int, step: int, steps_per_epoch: int) -> tuple[float, int, int]:
        """Run one training pass.

        Args:
            model: The unwrapped module, used for gradient clipping.
            forward: The module to call, possibly a compiled wrapper.
            optimizer: Optimiser to step.
            loader: Training batch source.
            mixer: Mixup transform, or None.
            epoch: Zero-based epoch number.
            step: Global optimiser step count on entry.
            steps_per_epoch: Optimiser steps per epoch, for the schedule.

        Returns:
            Mean loss, the updated step count, and the number of images seen.

        Raises:
            TrainingError: If the mean loss is not finite.
        """
        forward.train()
        loader.set_epoch(epoch)
        # [AUDIT T2] The running loss stays on the device. float(loss.detach()) on every
        # micro-batch forced a device-to-host sync per step, so the prefetching loader and
        # non_blocking copies could not overlap anything and images_per_second was understated.
        # Before:
        #   loss_total = 0.0
        loss_total = torch.zeros((), device=self.device)
        micro_batches = 0
        images_seen = 0
        optimizer.zero_grad(set_to_none=True)
        n_micro = len(loader)
        check_every = self.run.log_every or FINITE_CHECK_EVERY

        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            x = random_resized_crop_flip(xb, self.recipe.train_res, self.recipe.rrc_scale,
                                         self.recipe.rrc_ratio, self.recipe.hflip_prob)
            x = normalise(x, self.norm)
            if self.run.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)

            target = yb
            if mixer is not None:
                x, target = mixer(x, yb)

            with torch.autocast(self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp_dtype != torch.float32):
                logits = _as_logits(forward(x))
            logits = logits.float()

            if mixer is not None:
                # Mixup emits soft targets and has already applied label smoothing.
                loss = torch.sum(-target * F.log_softmax(logits, dim=-1), dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, target,
                                       label_smoothing=self.recipe.label_smoothing)

            # [AUDIT T1] The optimiser stepped only on (i + 1) % accum == 0, so trailing
            # micro-batches were computed and then discarded, and with limit_batches < accum it
            # never stepped at all. The final group of an epoch may now be partial: it is
            # stepped too, and its losses are divided by the group's real size so its gradient
            # is still a mean over the micro-batches it contains.
            # Before:
            #   (loss / self.accum).backward()
            #   ...
            #   if (i + 1) % self.accum == 0:
            group_start = (i // self.accum) * self.accum
            group_size = min(self.accum, n_micro - group_start)
            # Dividing by the group size makes accumulated gradients the mean over
            # micro-batches rather than the sum, which is what keeps the effective batch size
            # equivalent to a single large batch.
            (loss / group_size).backward()
            loss_total += loss.detach()
            micro_batches += 1
            images_seen += len(yb)

            if i + 1 == group_start + group_size:
                lr = lr_at_step(step, steps_per_epoch, self.recipe)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                if self.recipe.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.recipe.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            # [AUDIT T14/T15] Finiteness was checked only at the end of the epoch, so a
            # divergence in the first step wasted the whole epoch; log_every was never read.
            # Both now happen at one periodic sync.
            # Before: (no in-epoch check or progress line)
            if (i + 1) % check_every == 0:
                running = loss_total.item() / micro_batches
                if not math.isfinite(running):
                    raise TrainingError(
                        f"training loss became non-finite by micro-batch {i + 1} of epoch "
                        f"{epoch + 1}; lower the learning rate or enable gradient clipping")
                if self.run.log_every:
                    print(f"      step {i + 1}/{n_micro}  loss {running:.4f}  "
                          f"lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)

        mean_loss = loss_total.item() / max(1, micro_batches)
        if not math.isfinite(mean_loss):
            raise TrainingError(
                "training loss is not finite; lower the learning rate or enable gradient clipping")
        return mean_loss, step, images_seen

    @torch.inference_mode()
    def _evaluate(self, forward: nn.Module, loader: CacheLoader) -> tuple[float, float, float]:
        """Score the model on the validation cache.

        Args:
            forward: Module to call.
            loader: Validation batch source.

        Returns:
            Top-1 accuracy, top-5 accuracy and mean loss.

        Raises:
            TrainingError: If the loader yields no items.
        """
        forward.eval()
        # [AUDIT T2] Counts and loss stay on the device and are read once at the end; the
        # per-batch float() and int() conversions each forced a sync.
        # Before:
        #   c1 = c5 = seen = 0
        #   loss_sum = 0.0
        #   ...
        #   loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
        #   a, b = topk_hits(logits, yb)
        c1 = torch.zeros((), dtype=torch.long, device=self.device)
        c5 = torch.zeros((), dtype=torch.long, device=self.device)
        loss_sum = torch.zeros((), device=self.device)
        seen = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            # No augmentation: validation images were resized and cropped at cache-build time.
            x = normalise(centre_view(xb), self.norm)
            if self.run.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(self.device.type, dtype=self.amp_dtype,
                                enabled=self.amp_dtype != torch.float32):
                logits = _as_logits(forward(x))
            logits = logits.float()
            loss_sum += F.cross_entropy(logits, yb, reduction="sum")
            a, b = topk_hit_tensors(logits, yb)
            c1 += a
            c5 += b
            seen += len(yb)
        # [AUDIT T12] An empty validation set divided by zero.
        # Before: (no check)
        if seen == 0:
            raise TrainingError("validation loader yielded no items")
        return 100.0 * c1.item() / seen, 100.0 * c5.item() / seen, loss_sum.item() / seen

    # ---- resume -------------------------------------------------------------------------

    def _try_resume(self, model: nn.Module,
                    optimizer: torch.optim.Optimizer) -> dict[str, Any]:
        """Restore state from an interrupted run of the same recipe.

        A checkpoint from a different recipe is ignored rather than loaded: continuing one
        recipe from another's weights would produce a result attributable to neither.

        Args:
            model: Module to restore weights into.
            optimizer: Optimiser to restore state into.

        Returns:
            The restored train_state: at least "epoch", "step", "best_top1" and "best_epoch",
            plus "last_record" when the checkpoint carries one. Fresh-start values when there
            is nothing to resume.
        """
        fresh = {"epoch": 0, "step": 0, "best_top1": -1.0, "best_epoch": 0}
        if not (self.resume_path and self.resume_path.exists() and self.run.resume):
            return fresh
        payload = load_checkpoint(self.resume_path, map_location=str(self.device))
        meta = payload.get("meta") or {}
        if meta.get("recipe_digest") != self.recipe.digest():
            print("  resume checkpoint belongs to a different recipe; starting from scratch")
            return fresh
        model.load_state_dict(payload["state_dict"])
        if "optimizer" in payload:
            # Restoring the optimiser matters for AdamW: discarding the moment estimates
            # produces a visible transient in the loss at the resume point.
            optimizer.load_state_dict(payload["optimizer"])
        ts = {**fresh, **(payload.get("train_state") or {})}
        # [AUDIT T5] seed_everything() had reset every global RNG, so a resumed run replayed
        # epoch 0's crop, flip and mixup draws. The states saved with the checkpoint are now
        # restored, so a resumed run continues the exact streams of an uninterrupted one.
        # Before: (RNG states were neither saved nor restored)
        _set_rng_state(ts.get("rng"))
        print(f"  resuming from epoch {int(ts['epoch'])}")
        return ts

    def _completed_record(self, forward: nn.Module, val_loader: CacheLoader,
                          train_state: dict[str, Any],
                          optimizer: torch.optim.Optimizer) -> EpochRecord:
        """Build the final epoch record for a run whose checkpoint is already complete.

        Validation metrics are recomputed from the restored weights; training-side figures
        come from the record stored in the checkpoint, when it has one.

        Args:
            forward: Module to evaluate.
            val_loader: Validation batch source.
            train_state: Restored resume bookkeeping.
            optimizer: Restored optimiser, read for the final learning rate.

        Returns:
            The record describing the final epoch.
        """
        top1, top5, val_loss = self._evaluate(forward, val_loader)
        last = train_state.get("last_record") or {}
        print(f"  checkpoint already covers all {self.recipe.epochs} epochs; "
              f"reporting the restored weights")
        return EpochRecord(epoch=self.recipe.epochs,
                           train_loss=float(last.get("train_loss", float("nan"))),
                           val_loss=val_loss, top1=top1, top5=top5,
                           lr=float(last.get("lr", optimizer.param_groups[0]["lr"])),
                           epoch_seconds=float(last.get("epoch_seconds", 0.0)),
                           images_per_second=float(last.get("images_per_second", 0.0)))

    # ---- entry point --------------------------------------------------------------------

    def fit(self) -> TrainingResult:
        """Train the model to completion.

        Returns:
            The training outcome, including per-epoch history for this session.

        Raises:
            TrainingError: If the architecture cannot be resolved or training diverges.
        """
        seed_everything(self.recipe.seed)
        # [AUDIT T15] Input shapes are fixed for the whole run, which is the case cuDNN
        # autotuning is for. It was described in seed_everything's NOTE but never enabled.
        # Before: (not set; cuDNN default is False)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        resolved = self.registry.resolve(self.spec, self.recipe.num_classes)
        model = resolved.module.to(self.device)
        if self.run.channels_last:
            model = model.to(memory_format=torch.channels_last)

        # Two references: `forward` is what gets called and may be a compiled wrapper, while
        # `model` stays unwrapped so its state dictionary keys remain clean.
        forward = torch.compile(model) if self.run.compile else model
        optimizer = build_optimizer(model, self.recipe)
        mixer = self._make_mixer()

        train_loader = CacheLoader(self.train_cache, self.micro_bs, shuffle=True,
                                   seed=self.recipe.seed, prefetch=self.run.prefetch,
                                   limit_batches=self.run.limit_batches,
                                   pin_memory=self.device.type == "cuda")
        # [AUDIT T13] limit_batches used to cap validation too, and the val cache is in
        # class-sorted order, so a smoke run scored only the first few classes. A limited run
        # now validates on a fixed random subset of the same size.
        # Before:
        #   val_loader = CacheLoader(self.val_cache, max(1, self.micro_bs // 2), shuffle=False,
        #                            prefetch=self.run.prefetch,
        #                            limit_batches=self.run.limit_batches,
        #                            pin_memory=self.device.type == "cuda")
        val_bs = max(1, self.micro_bs // 2)
        val_indices = None
        if self.run.limit_batches:
            val_indices = random_subset(len(self.val_cache), self.run.limit_batches * val_bs,
                                        seed=self.recipe.seed)
        val_loader = CacheLoader(self.val_cache, val_bs, shuffle=False,
                                 prefetch=self.run.prefetch, indices=val_indices,
                                 pin_memory=self.device.type == "cuda")
        # [AUDIT T1] The final partial accumulation group is stepped, so it counts.
        # Before:
        #   steps_per_epoch = max(1, len(train_loader) // self.accum)
        steps_per_epoch = math.ceil(len(train_loader) / self.accum)

        # Before:
        #   start_epoch, step, best_top1, best_epoch = self._try_resume(model, optimizer)
        ts = self._try_resume(model, optimizer)
        start_epoch, step = int(ts["epoch"]), int(ts["step"])
        best_top1, best_epoch = float(ts["best_top1"]), int(ts["best_epoch"])

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        history: list[EpochRecord] = []
        session_seconds = 0.0

        # [AUDIT T4] A crash after the last epoch's checkpoint but before the final save left
        # start_epoch == epochs; fit() then raised "nothing to train" on every rerun and the
        # model was permanently marked failed. The restored weights are now evaluated and
        # reported as the finished result.
        # Before: (the loop below simply did not run, and the "if not history" check raised)
        if start_epoch >= self.recipe.epochs:
            final = self._completed_record(forward, val_loader, ts, optimizer)
            resolved.module = model
            return TrainingResult(
                model=self.spec.name, resolved=resolved,
                epochs_completed=self.recipe.epochs, final=final,
                best_top1=max(best_top1, final.top1), best_epoch=best_epoch or final.epoch,
                train_minutes=0.0, seconds_per_epoch=0.0, peak_memory_mb=None, history=[])

        for epoch in range(start_epoch, self.recipe.epochs):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_loss, step, seen = self._train_epoch(
                model, forward, optimizer, train_loader, mixer, epoch, step, steps_per_epoch)
            if self.device.type == "cuda":
                # CUDA launches are asynchronous; without this the timer measures queueing.
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            session_seconds += elapsed

            top1, top5, val_loss = self._evaluate(forward, val_loader)
            if top1 > best_top1:
                best_top1, best_epoch = top1, epoch + 1

            record = EpochRecord(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 top1=top1, top5=top5, lr=optimizer.param_groups[0]["lr"],
                                 epoch_seconds=round(elapsed, 2),
                                 images_per_second=round(seen / elapsed, 1))
            history.append(record)
            if self.on_epoch:
                self.on_epoch(record)

            # [AUDIT T5/T4/T9] The checkpoint now also carries the RNG states (T5), the last
            # epoch record so a final-epoch resume can report it (T4), and the effective kwargs
            # the registry built with, including any forced ones (T9).
            # Before:
            #   kwargs=dict(self.spec.kwargs),
            #   ...
            #   train_state={"epoch": epoch + 1, "step": step,
            #                "best_top1": best_top1, "best_epoch": best_epoch})
            if self.resume_path and (epoch + 1) % self.run.checkpoint_every == 0:
                save_checkpoint(
                    self.resume_path, model,
                    CheckpointMeta(name=self.spec.name, source=resolved.source,
                                   provider_version=resolved.provider_version,
                                   kwargs=dict(resolved.kwargs),
                                   num_classes=self.recipe.num_classes,
                                   classes=list(self.train_cache.classes),
                                   recipe_digest=self.recipe.digest(),
                                   epochs_completed=epoch + 1),
                    self.recipe, optimizer=optimizer,
                    train_state={"epoch": epoch + 1, "step": step,
                                 "best_top1": best_top1, "best_epoch": best_epoch,
                                 "last_record": asdict(record), "rng": _get_rng_state()})

        peak_mb = (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                   if self.device.type == "cuda" else None)
        resolved.module = model
        return TrainingResult(
            model=self.spec.name, resolved=resolved,
            epochs_completed=self.recipe.epochs, final=history[-1],
            best_top1=best_top1, best_epoch=best_epoch,
            train_minutes=round(session_seconds / 60, 2),
            seconds_per_epoch=round(session_seconds / len(history), 1),
            peak_memory_mb=peak_mb, history=history)


# TO DO: Implement Custom classes that implement training loops for other tasks: Detection, Face Detection, Recognition,
# Tracking,Segmentation.