"""
FILENAME:       src/edgebench/core/artifacts.py
DESCRIPTION:    Checkpoint writing and reading. A checkpoint carries enough provenance to
                rebuild its architecture and reproduce its evaluation without consulting the
                code that produced it: the resolved implementation and its version, the recipe
                digest, the class list and the full recipe. This is what allows the export and
                benchmark stages to operate on a checkpoint alone.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from edgebench.core.config import RecipeConfig
from edgebench.core.registry import ModelRegistry, ModelSpec, get_registry

# Format version of the checkpoint dictionary. Incremented when the key layout changes, so a
# reader can refuse an incompatible file rather than fail obscurely on a missing key.
CHECKPOINT_VERSION: int = 1

# Prefixes added by module wrappers. Stripped on load so a checkpoint saved from a wrapped
# model still loads into a bare one.
WRAPPER_PREFIXES: tuple[str, ...] = ("_orig_mod.", "module.")


class ArtifactError(RuntimeError):
    """Raised when a checkpoint is malformed, incompatible, or cannot be reconstructed."""


@dataclass
class CheckpointMeta:
    """Provenance describing one trained model.

    Attributes:
        name: Architecture identifier.
        source: Provider that supplied the implementation.
        provider_version: Version of that provider, or repository and commit for an external
            implementation.
        kwargs: Constructor arguments used to build the architecture.
        num_classes: Size of the classifier output.
        classes: Class names indexed by label value.
        recipe_digest: Digest of the recipe the model was trained under.
        task: Task identifier.
        dataset: Dataset identifier.
        epochs_completed: Epochs actually trained.
        metrics: Final evaluation metrics.
        extra: Additional free-form fields.
    """

    name: str
    source: str = ""
    provider_version: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    num_classes: int = 0
    classes: list[str] = field(default_factory=list)
    recipe_digest: str = ""
    task: str = ""
    dataset: str = ""
    epochs_completed: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for storage.

        Returns:
            Mapping of field name to value.
        """
        return {
            "name": self.name, "source": self.source,
            "provider_version": self.provider_version, "kwargs": dict(self.kwargs),
            "num_classes": self.num_classes, "classes": list(self.classes),
            "recipe_digest": self.recipe_digest, "task": self.task, "dataset": self.dataset,
            "epochs_completed": self.epochs_completed, "metrics": dict(self.metrics),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointMeta":
        """Construct from a mapping, tolerating unknown keys.

        Unknown keys are ignored rather than rejected: a checkpoint written by a later release
        must remain loadable.

        Args:
            data: Mapping of field name to value.

        Returns:
            The constructed metadata.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def model_spec(self) -> ModelSpec:
        """Rebuild the specification that produced this model.

        Returns:
            A ModelSpec pinned to the recorded provider, so reconstruction cannot silently
            resolve to a different implementation than the one that was trained.
        """
        return ModelSpec(name=self.name, source=self.source or None, kwargs=dict(self.kwargs))


def strip_wrapper_prefixes(state: Mapping[str, Any]) -> dict[str, Any]:
    """Remove a uniform wrapper prefix from every key of a state dictionary.

    torch.compile and the distributed wrappers rename every parameter. Stripping lets a
    checkpoint saved through a wrapper load into a bare module.

    Args:
        state: State dictionary.

    Returns:
        The state dictionary with any single uniform prefix removed.
    """
    # [AUDIT A1] A single pass in fixed order left "_orig_mod." on DDP(compile(model)) keys
    # ("module._orig_mod.fc.weight" -> "_orig_mod.fc.weight"). Strip until nothing changes.
    # Before:
    #   out = dict(state)
    #   for prefix in WRAPPER_PREFIXES:
    #       if out and all(k.startswith(prefix) for k in out):
    #           out = {k[len(prefix):]: v for k, v in out.items()}
    #   return out
    out = dict(state)
    stripped = True
    while out and stripped:
        stripped = False
        for prefix in WRAPPER_PREFIXES:
            if all(k.startswith(prefix) for k in out):
                out = {k[len(prefix):]: v for k, v in out.items()}
                stripped = True
    return out


def _is_state_dict(candidate: Any) -> bool:
    """Report whether a mapping looks like a module state dictionary.

    Args:
        candidate: Object to test.

    Returns:
        True when it is a non-empty mapping of tensors, allowing the non-tensor values that
        modules store under keys ending in "_extra_state".
    """
    if not isinstance(candidate, Mapping) or not candidate:
        return False
    tensors = 0
    for k, v in candidate.items():
        if torch.is_tensor(v):
            tensors += 1
        elif not str(k).endswith("_extra_state"):
            return False
    return tensors > 0


def extract_state_dict(obj: Any) -> dict[str, Any]:
    """Locate the weight tensors inside a loaded checkpoint object.

    Supports this framework's own layout and the bare state dictionaries that third-party
    repositories commonly publish.

    Args:
        obj: Object returned by torch.load.

    Returns:
        The state dictionary.

    Raises:
        ArtifactError: If no tensor mapping can be found.
    """
    # [AUDIT A2] Every value had to be a tensor, which rejected modules that store
    # "_extra_state" entries.
    # Before:
    #   if isinstance(candidate, Mapping) and candidate and \
    #           all(torch.is_tensor(v) for v in candidate.values()):
    #   ...
    #   if obj and all(torch.is_tensor(v) for v in obj.values()):
    if isinstance(obj, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            candidate = obj.get(key)
            if _is_state_dict(candidate):
                return dict(candidate)
        if _is_state_dict(obj):
            return dict(obj)
    raise ArtifactError(
        f"no state dictionary found; top-level keys: {list(obj)[:10] if isinstance(obj, Mapping) else type(obj)}")


def infer_num_classes(state: Mapping[str, Any]) -> int | None:
    """Infer the classifier width from the final weight tensor.

    Needed for third-party checkpoints that carry no metadata. The classifier is the last
    weight tensor in every architecture in scope, and its leading dimension is the class count
    for both a linear head and the 1x1 convolution head SqueezeNet uses.

    Args:
        state: State dictionary.

    Returns:
        The inferred class count, or None when no suitable tensor is present.
    """
    key = classifier_weight_key(state)
    return None if key is None else int(state[key].shape[0])


def classifier_weight_key(state: Mapping[str, Any]) -> str | None:
    """Locate the tensor infer_num_classes reads the class count from.

    Args:
        state: State dictionary.

    Returns:
        The key of the last weight tensor with at least two dimensions, or None.
    """
    for key in reversed(list(state)):
        tensor = state[key]
        if torch.is_tensor(tensor) and tensor.ndim >= 2 and key.endswith("weight"):
            return key
    return None


def save_checkpoint(path: "str | Path",
                    model: nn.Module,
                    meta: CheckpointMeta,
                    recipe: RecipeConfig,
                    *,
                    optimizer: torch.optim.Optimizer | None = None,
                    train_state: Mapping[str, Any] | None = None) -> Path:
    """Write a checkpoint atomically.

    Args:
        path: Destination path.
        model: Module whose weights to store. Pass the unwrapped module, not a compiled or
            distributed wrapper, so the stored keys need no stripping on load.
        meta: Provenance describing the model.
        recipe: Recipe the model was trained under, stored in full so the training conditions
            are recoverable from the checkpoint alone.
        optimizer: Optimiser whose state to store. Required for a resumable checkpoint, since
            restarting AdamW without its moment estimates perturbs training.
        train_state: Resume bookkeeping such as the epoch and step counters.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "meta": meta.to_dict(),
        "recipe": recipe.to_dict(),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if train_state:
        payload["train_state"] = dict(train_state)

    # Written to a temporary file and renamed: a checkpoint interrupted mid-write would
    # otherwise be reported as present by the skip rule and fail only when loaded.
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: "str | Path", map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint file.

    SAFETY: unpickling executes arbitrary code, so weights_only is disabled only because this
    framework's checkpoints carry metadata that is not tensors. Apply this to files produced
    locally; treat downloaded weights as untrusted and load them with weights_only=True.

    Args:
        path: Checkpoint path.
        map_location: Device to map tensors onto.

    Returns:
        The loaded payload.

    Raises:
        ArtifactError: If the file is missing or its format version is newer than supported.
    """
    p = Path(path)
    if not p.is_file():
        raise ArtifactError(f"checkpoint not found: {p}")
    obj = torch.load(p, map_location=map_location, weights_only=False)
    if isinstance(obj, Mapping):
        version = obj.get("version", 0)
        if isinstance(version, int) and version > CHECKPOINT_VERSION:
            raise ArtifactError(
                f"{p} uses checkpoint format v{version}, this build supports v{CHECKPOINT_VERSION}")
    return obj if isinstance(obj, dict) else {"state_dict": obj}


def read_meta(payload: Mapping[str, Any], fallback_name: str = "") -> CheckpointMeta:
    """Extract provenance from a loaded checkpoint.

    Args:
        payload: Loaded checkpoint.
        fallback_name: Architecture name to assume when the checkpoint records none, normally
            the filename stem. Third-party weights usually have no metadata.

    Returns:
        The provenance, with the class count inferred from the weights when unrecorded.
    """
    meta = CheckpointMeta.from_dict(payload.get("meta") or {})
    if not meta.name:
        meta.name = fallback_name
    # [AUDIT A3] The inference is a heuristic; the tensor it chose is now printed so a wrong
    # guess is visible.
    # Before:
    #   state = strip_wrapper_prefixes(extract_state_dict(payload))
    #   meta.num_classes = infer_num_classes(state) or 0
    if not meta.num_classes:
        state = strip_wrapper_prefixes(extract_state_dict(payload))
        meta.num_classes = infer_num_classes(state) or 0
        if meta.num_classes:
            print(f"  NOTE: inferred num_classes={meta.num_classes} for {meta.name!r} "
                  f"from {classifier_weight_key(state)!r}")
    return meta


def read_recipe(payload: Mapping[str, Any]) -> RecipeConfig | None:
    """Extract the training recipe from a loaded checkpoint.

    Args:
        payload: Loaded checkpoint.

    Returns:
        The recipe, or None when the checkpoint carries none or an unreadable one.
    """
    data = payload.get("recipe")
    if not isinstance(data, Mapping):
        return None
    try:
        return RecipeConfig.from_dict(data)
    except Exception:
        # A recipe written by a different schema version must not prevent the weights from
        # being used; the caller decides whether the missing recipe matters.
        return None


def rebuild_model(payload: Mapping[str, Any],
                  fallback_name: str = "",
                  registry: ModelRegistry | None = None,
                  strict: bool = True) -> tuple[nn.Module, CheckpointMeta]:
    """Reconstruct a trained model from a loaded checkpoint.

    Args:
        payload: Loaded checkpoint.
        fallback_name: Architecture name to assume when the checkpoint records none.
        registry: Registry to resolve the architecture through. Defaults to the shared one.
        strict: Require an exact match between checkpoint keys and module parameters. Kept on
            by default: a tolerated mismatch produces a network with partly random weights,
            which exports and benchmarks without complaint and yields meaningless accuracy.

    Returns:
        The module in evaluation mode, and its provenance.

    Raises:
        ArtifactError: If the architecture cannot be resolved or the weights do not fit it.
    """
    meta = read_meta(payload, fallback_name)
    if not meta.name:
        raise ArtifactError("checkpoint records no architecture name and none was supplied")
    if not meta.num_classes:
        raise ArtifactError(f"cannot determine num_classes for {meta.name!r}")

    reg = registry or get_registry()
    try:
        resolved = reg.resolve(meta.model_spec(), meta.num_classes)
    except Exception as exc:
        raise ArtifactError(f"cannot rebuild {meta.name!r}: {exc}") from exc

    # A provider upgrade can change an architecture's definition. The weights may still load,
    # so this is a warning rather than an error, but it must be visible.
    if meta.provider_version and resolved.provider_version != meta.provider_version:
        print(f"  NOTE: {meta.name} was trained with {meta.source} "
              f"{meta.provider_version}, rebuilding with {resolved.provider_version}")

    state = strip_wrapper_prefixes(extract_state_dict(payload))
    try:
        resolved.module.load_state_dict(state, strict=strict)
    except RuntimeError as exc:
        raise ArtifactError(f"weights do not fit {meta.name!r}: {exc}") from exc

    # eval() before returning: BatchNorm must use its running statistics and dropout must be
    # inert, both for inference and for a correct ONNX export.
    resolved.module.eval()
    return resolved.module, meta