"""
FILENAME:       src/edgexbench/models/external/shufflenetv1.py
DESCRIPTION:    Registers ShuffleNetV1 from the vendored Megvii implementation. Neither timm nor
                torchvision ships V1, so the lineage would otherwise be represented by V2 alone.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-29
"""

from __future__ import annotations

from torch import nn

from edgexbench.core.registry import get_registry

REPO = "megvii-model/ShuffleNet-Series"
COMMIT = "aa91feb71b01f28d0b8da3533d20a3edb11b1810"
PAPER = "Zhang et al., CVPR 2018, arXiv:1707.01083"

# Width multipliers and group counts reported in the upstream README.
VARIANTS: tuple[tuple[str, int], ...] = (
    ("0.5x", 3), ("1.0x", 3), ("1.5x", 3), ("2.0x", 3),
    ("1.0x", 8),
)


def build_shufflenetv1(num_classes: int, model_size: str = "1.0x",
                       group: int = 3, **_: object) -> nn.Module:
    """Build one ShuffleNetV1 variant.

    Args:
        num_classes: Size of the classifier output.
        model_size: Width multiplier, one of "0.5x", "1.0x", "1.5x", "2.0x".
        group: Groups in the pointwise convolutions; upstream supports 3 and 8.

    Returns:
        The untrained module.
    """
    # Imported here so a broken vendored repo breaks only this architecture, not the sweep.
    from ._vendor.shufflenetv1.network import ShuffleNetV1 
    return ShuffleNetV1(n_class=num_classes, model_size=model_size, group=group)


def _variant_name(model_size: str, group: int) -> str:
    """Build the registry name for a variant.

    Width and group count are part of the identifier because they are architectural variants
    that may be trained alongside each other.

    Args:
        model_size: Width multiplier.
        group: Group count.

    Returns:
        A filesystem-safe model identifier.
    """
    return f"shufflenetv1_{model_size.replace('.', '_')}_g{group}"


def register() -> None:
    """Register every ShuffleNetV1 variant from this repository."""
    registry = get_registry()
    for model_size, group in VARIANTS:
        registry.register_external(
            _variant_name(model_size, group),
            lambda num_classes, _ms=model_size, _g=group, **kw:
                build_shufflenetv1(num_classes, model_size=_ms, group=_g, **kw),
            repo=REPO, commit=COMMIT, paper=PAPER,
            notes="Channel shuffle is reshape/transpose/reshape plus grouped 1x1 convolutions; "
                  "a deployability test case for fixed-function accelerators.",
        )