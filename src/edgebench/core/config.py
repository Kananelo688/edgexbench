"""
FILENAME:       src/edgebench/core/config.py
DESCRIPTION:    Configuration objects for training runs, split into a hashed recipe and an
                unhashed execution config. The split is structural: every field of RecipeConfig
                affects the trained weights and is therefore included in the recipe digest,
                while RunConfig holds only execution settings that cannot. This removes the
                need for a manually maintained hash-exclusion list, which is a silent source of
                invalid provenance when a new field is added and forgotten.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Mapping

# Mixed-precision dtypes accepted by the trainer. bfloat16 is preferred on Ampere and later:
# it carries the fp32 exponent range, so training needs no gradient scaler.
AMP_DTYPES: tuple[str, ...] = ("float32", "float16", "bfloat16")

OPTIMIZERS: tuple[str, ...] = ("adamw", "sgd")
SCHEDULERS: tuple[str, ...] = ("cosine", "step", "constant")

# Length of the recipe digest. 12 hex characters is ample to distinguish the recipes one project
# will ever hold, while remaining short enough to read in a filename or a CSV cell.
DIGEST_LENGTH: int = 12


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent or malformed."""


def _coerce_pair(value: Any, name: str) -> tuple[float, float]:
    """Convert a two-element sequence to a tuple of floats.

    YAML and JSON both decode a pair as a list, which would make two otherwise identical
    configurations hash differently depending on their source. Coercing at construction time
    normalises that.

    Args:
        value: Candidate pair.
        name: Field name, used in the error message.

    Returns:
        The pair as a tuple of floats.

    Raises:
        ConfigError: If the value is not a two-element sequence of numbers.
    """
    try:
        lo, hi = value
        return (float(lo), float(hi))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a two-element sequence, got {value!r}") from exc


@dataclass(frozen=True)
class RecipeConfig:
    """Training settings that determine the trained weights.

    Every field here is hashed. Two runs sharing a digest were trained under identical
    conditions, which is the claim a cross-architecture comparison depends on. Anything that
    cannot change the weights belongs in RunConfig instead.

    NOTE: Adding a field to this class changes the digest of every existing recipe, even when
    the new field's default reproduces the previous behaviour. This is deliberate and fails
    closed: a stale result is marked for recomputation rather than silently accepted.

    Attributes:
        num_classes: Size of the classifier output.
        train_res: Resolution images are trained at.
        eval_res: Resolution images are evaluated at. Training below evaluation resolution is a
            budget technique that also exploits the train-test resolution discrepancy
            (Touvron et al., 2019).
        eval_crop_pct: Centre-crop ratio for evaluation; images are resized to
            eval_res / eval_crop_pct before cropping.
        epochs: Total training epochs.
        batch_size: Effective batch size. Gradient accumulation keeps this constant across
            models regardless of what fits in memory.
        optimizer: Optimiser identifier.
        lr: Peak learning rate, paired with batch_size.
        min_lr: Floor of the decay schedule.
        scheduler: Learning-rate schedule identifier.
        warmup_epochs: Epochs of linear warmup before the schedule begins.
        weight_decay: Decay applied to convolution and linear weights only, never to
            normalisation parameters or biases.
        momentum: SGD momentum. Ignored by AdamW but hashed regardless, so a recipe file
            remains a complete description of itself.
        label_smoothing: Cross-entropy label smoothing.
        mixup_alpha: Mixup beta parameter; 0 disables mixup.
        cutmix_alpha: CutMix beta parameter; 0 disables CutMix.
        mix_prob: Fraction of batches to which mixup or CutMix is applied.
        rrc_scale: Random-resized-crop area range, as a fraction of the source image.
        rrc_ratio: Random-resized-crop aspect-ratio range.
        hflip_prob: Horizontal flip probability.
        clip_grad: Gradient-norm clipping threshold; 0 disables clipping.
        seed: Seed for weight initialisation and data ordering.
        amp_dtype: Autocast dtype for the forward pass.
    """

    # --- data ---
    num_classes: int = 100
    train_res: int = 160
    eval_res: int = 224
    eval_crop_pct: float = 0.875

    # --- schedule ---
    epochs: int = 100
    batch_size: int = 256
    optimizer: str = "adamw"
    lr: float = 2e-3
    min_lr: float = 1e-5
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    momentum: float = 0.9

    # --- regularisation ---
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.1
    cutmix_alpha: float = 1.0
    mix_prob: float = 0.5
    rrc_scale: tuple[float, float] = (0.35, 1.0)
    rrc_ratio: tuple[float, float] = (3 / 4, 4 / 3)
    hflip_prob: float = 0.5

    # --- numerics ---
    clip_grad: float = 5.0
    seed: int = 42
    amp_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        """Normalise sequence fields and validate the configuration.

        Returns:
            None.

        Raises:
            ConfigError: If any field is out of range or the combination is inconsistent.
        """
        object.__setattr__(self, "rrc_scale", _coerce_pair(self.rrc_scale, "rrc_scale"))
        object.__setattr__(self, "rrc_ratio", _coerce_pair(self.rrc_ratio, "rrc_ratio"))
        self._validate()

    def _validate(self) -> None:
        """Check every field and cross-field relationship.

        Returns:
            None.

        Raises:
            ConfigError: If the configuration is invalid.
        """
        err: list[str] = []

        if self.num_classes <= 0:
            err.append(f"num_classes must be positive, got {self.num_classes}")
        for name in ("train_res", "eval_res", "epochs", "batch_size"):
            if getattr(self, name) <= 0:
                err.append(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 < self.eval_crop_pct <= 1:
            err.append(f"eval_crop_pct must be in (0, 1], got {self.eval_crop_pct}")

        if self.optimizer not in OPTIMIZERS:
            err.append(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.scheduler not in SCHEDULERS:
            err.append(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.amp_dtype not in AMP_DTYPES:
            err.append(f"amp_dtype must be one of {AMP_DTYPES}, got {self.amp_dtype!r}")

        if self.lr <= 0:
            err.append(f"lr must be positive, got {self.lr}")
        if self.min_lr < 0:
            err.append(f"min_lr must be non-negative, got {self.min_lr}")
        if self.min_lr >= self.lr:
            err.append(f"min_lr ({self.min_lr}) must be below lr ({self.lr})")
        if self.warmup_epochs < 0:
            err.append(f"warmup_epochs must be non-negative, got {self.warmup_epochs}")
        # Warmup consuming the whole run would leave no decay phase, which is almost certainly
        # a configuration error rather than an intent.
        if self.warmup_epochs >= self.epochs:
            err.append(f"warmup_epochs ({self.warmup_epochs}) must be below "
                       f"epochs ({self.epochs})")
        if self.weight_decay < 0:
            err.append(f"weight_decay must be non-negative, got {self.weight_decay}")
        if not 0 <= self.momentum < 1:
            err.append(f"momentum must be in [0, 1), got {self.momentum}")

        if not 0 <= self.label_smoothing < 1:
            err.append(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        for name in ("mixup_alpha", "cutmix_alpha"):
            if getattr(self, name) < 0:
                err.append(f"{name} must be non-negative, got {getattr(self, name)}")
        for name in ("mix_prob", "hflip_prob"):
            if not 0 <= getattr(self, name) <= 1:
                err.append(f"{name} must be in [0, 1], got {getattr(self, name)}")
        if self.clip_grad < 0:
            err.append(f"clip_grad must be non-negative, got {self.clip_grad}")

        lo, hi = self.rrc_scale
        if not 0 < lo <= hi <= 1:
            err.append(f"rrc_scale must satisfy 0 < lo <= hi <= 1, got {self.rrc_scale}")
        lo, hi = self.rrc_ratio
        if not 0 < lo <= hi:
            err.append(f"rrc_ratio must satisfy 0 < lo <= hi, got {self.rrc_ratio}")

        if err:
            raise ConfigError("invalid RecipeConfig:\n  - " + "\n  - ".join(err))

    @property
    def mix_enabled(self) -> bool:
        """Report whether mixup or CutMix is active.

        Returns:
            True when either alpha is non-zero and mixing is applied to some batches.
        """
        return (self.mixup_alpha > 0 or self.cutmix_alpha > 0) and self.mix_prob > 0

    @property
    def eval_resize(self) -> int:
        """Short-side resize applied before the evaluation centre crop.

        Returns:
            The resize target in pixels.
        """
        return int(round(self.eval_res / self.eval_crop_pct))

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Returns:
            Mapping of field name to value, with tuples converted to lists so the result is
            directly JSON- and YAML-serialisable.
        """
        out = asdict(self)
        for k, v in out.items():
            if isinstance(v, tuple):
                out[k] = list(v)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RecipeConfig":
        """Construct from a mapping, rejecting unrecognised keys.

        Unknown keys are an error rather than being ignored: silently dropping a misspelled
        field would train under a different recipe than the file describes.

        Args:
            data: Mapping of field name to value.

        Returns:
            The constructed configuration.

        Raises:
            ConfigError: If the mapping carries unknown keys or invalid values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown RecipeConfig fields {sorted(unknown)}; known: {sorted(known)}")
        return cls(**dict(data))

    def digest(self) -> str:
        """Compute the recipe digest.

        Serialised with sorted keys so the digest depends only on the values, never on field
        declaration order or the order keys appeared in a file.

        Returns:
            The first DIGEST_LENGTH hex characters of the SHA-256 of the canonical form.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:DIGEST_LENGTH]

    def derive(self, **changes: Any) -> "RecipeConfig":
        """Create a modified copy.

        Args:
            **changes: Field values to override.

        Returns:
            A new validated configuration with a different digest.

        Raises:
            ConfigError: If the resulting configuration is invalid.
        """
        return replace(self, **changes)

    def summary(self) -> str:
        """Render a one-line human-readable summary for run logs.

        Returns:
            A compact description including the digest.
        """
        return (f"[{self.digest()}] {self.optimizer} lr={self.lr:g} wd={self.weight_decay:g} "
                f"bs={self.batch_size} ep={self.epochs} warmup={self.warmup_epochs} "
                f"{self.train_res}->{self.eval_res}px {self.amp_dtype}")


@dataclass
class RunConfig:
    """Execution settings that cannot change the trained weights.

    Excluded from the recipe digest by construction. Mutable, unlike RecipeConfig, because
    these are routinely adjusted between runs of the same recipe.

    NOTE: micro_batch_size is here rather than in RecipeConfig because gradient accumulation
    keeps the effective batch size fixed. Normalisation statistics are still computed per
    micro-batch, so the equivalence is close but not exact; record the value used when
    reporting, and keep it constant for any model whose result will be compared directly.

    Attributes:
        device: Torch device string, or "auto" to select CUDA when available.
        micro_batch_size: Per-step batch when the effective batch does not fit in memory.
            None uses the recipe's batch size directly.
        compile: Whether to wrap the model with torch.compile.
        channels_last: Whether to use the channels-last memory format, which suits tensor-core
            convolution kernels.
        workers: Loader worker count, where the loader supports it.
        prefetch: Batches prepared ahead of the device.
        limit_batches: Cap on batches per epoch, for smoke tests only.
        debug: Route outputs to isolated debug directories so trial runs cannot contaminate
            recorded results.
        resume: Whether to continue from an interrupted run of the same recipe.
        checkpoint_every: Epoch interval between resume checkpoints.
        log_every: Step interval between progress lines; 0 logs once per epoch.
        output_tag: Optional suffix distinguishing concurrent runs of one recipe.
    """

    device: str = "auto"
    micro_batch_size: int | None = None
    compile: bool = False
    channels_last: bool = True
    workers: int = 4
    prefetch: int = 4
    limit_batches: int | None = None
    debug: bool = False
    resume: bool = True
    checkpoint_every: int = 1
    log_every: int = 0
    output_tag: str | None = None

    def __post_init__(self) -> None:
        """Validate the execution settings.

        Returns:
            None.

        Raises:
            ConfigError: If any value is out of range.
        """
        err: list[str] = []
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            err.append(f"micro_batch_size must be positive or None, got {self.micro_batch_size}")
        if self.workers < 0:
            err.append(f"workers must be non-negative, got {self.workers}")
        if self.prefetch < 1:
            err.append(f"prefetch must be at least 1, got {self.prefetch}")
        if self.limit_batches is not None and self.limit_batches <= 0:
            err.append(f"limit_batches must be positive or None, got {self.limit_batches}")
        if self.checkpoint_every < 1:
            err.append(f"checkpoint_every must be at least 1, got {self.checkpoint_every}")
        if self.log_every < 0:
            err.append(f"log_every must be non-negative, got {self.log_every}")
        if err:
            raise ConfigError("invalid RunConfig:\n  - " + "\n  - ".join(err))

    def accumulation_steps(self, recipe: RecipeConfig) -> int:
        """Compute the gradient-accumulation factor for a recipe.

        Args:
            recipe: The recipe supplying the effective batch size.

        Returns:
            Number of micro-batches per optimiser step.

        Raises:
            ConfigError: If the effective batch size is not divisible by micro_batch_size.
                An inexact split would change the effective batch and therefore the recipe.
        """
        if self.micro_batch_size is None:
            return 1
        if recipe.batch_size % self.micro_batch_size:
            raise ConfigError(
                f"batch_size ({recipe.batch_size}) must be divisible by micro_batch_size "
                f"({self.micro_batch_size}); otherwise the effective batch size changes")
        return recipe.batch_size // self.micro_batch_size

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary.

        Returns:
            Mapping of field name to value.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        """Construct from a mapping, rejecting unrecognised keys.

        Args:
            data: Mapping of field name to value.

        Returns:
            The constructed configuration.

        Raises:
            ConfigError: If the mapping carries unknown keys or invalid values.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown RunConfig fields {sorted(unknown)}; known: {sorted(known)}")
        return cls(**dict(data))


@dataclass
class ExperimentConfig:
    """A recipe, its execution settings, and the model list they apply to.

    Attributes:
        task: Task identifier, used to resolve output directories.
        recipe: The hashed training recipe.
        run: Execution settings.
        models: Model specifications, in any form accepted by ModelSpec.coerce. Left
            unparsed here so this module need not import the registry.
        dataset: Dataset identifier used to locate caches.
        name: Optional human-readable label for the experiment.
    """

    task: str = "classification"
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    run: RunConfig = field(default_factory=RunConfig)
    models: list[Any] = field(default_factory=list)
    dataset: str = "imageNet-100"
    name: str | None = None

    @property
    def digest(self) -> str:
        """Recipe digest for this experiment.

        Returns:
            The recipe digest. Deliberately excludes the model list: adding a model to an
            experiment must not invalidate the models already trained under that recipe.
        """
        return self.recipe.digest()

    def to_dict(self) -> dict[str, Any]:
        """Serialise the whole experiment.

        Returns:
            Nested mapping suitable for writing to YAML.
        """
        return {
            "task": self.task,
            "name": self.name,
            "dataset": self.dataset,
            "recipe": self.recipe.to_dict(),
            "run": self.run.to_dict(),
            "models": list(self.models),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        """Construct from a nested mapping.

        Args:
            data: Mapping with optional "recipe", "run" and "models" sections.

        Returns:
            The constructed experiment configuration.

        Raises:
            ConfigError: If the mapping carries unknown top-level keys or invalid sections.
        """
        known = {"task", "name", "dataset", "recipe", "run", "models"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown ExperimentConfig fields {sorted(unknown)}; known: {sorted(known)}")
        return cls(
            task=data.get("task", "classification"),
            name=data.get("name"),
            dataset=data.get("dataset", "imageNet-100"),
            recipe=RecipeConfig.from_dict(data.get("recipe") or {}),
            run=RunConfig.from_dict(data.get("run") or {}),
            models=list(data.get("models") or []),
        )

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "ExperimentConfig":
        """Load an experiment from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            The constructed experiment configuration.

        Raises:
            ConfigError: If the file is missing, unreadable, or does not contain a mapping.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is required to read config files: pip install pyyaml") from exc
        try:
            data = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"could not parse {p}: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ConfigError(f"{p} must contain a mapping at the top level")
        return cls.from_dict(data)

    def to_yaml(self, path: "str | Path") -> Path:
        """Write the experiment to a YAML file.

        Args:
            path: Destination path.

        Returns:
            The path written.

        Raises:
            ConfigError: If PyYAML is unavailable.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is required to write config files: pip install pyyaml") from exc
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys=False preserves the declaration order, which groups related settings
        # together and keeps a hand-edited file readable after a round trip.
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False))
        return p