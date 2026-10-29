"""Shared fixtures for the test suite: an in-memory cache and a tiny registered model."""

from __future__ import annotations

import numpy as np
from torch import nn

from edgexbench.core.registry import ModelRegistry


class FakeCache:
    """In-memory stand-in for ImageCache with the same read interface."""

    def __init__(self, n: int, res: int, n_classes: int = 4, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, (n, res, res, 3), dtype=np.uint8)
        self.labels = (np.arange(n) % n_classes).astype(np.int64)
        self.resolution = res
        self.classes = [f"c{i}" for i in range(n_classes)]

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, indices, out=None):
        idx = np.asarray(indices)
        if out is not None:
            np.take(self.images, idx, axis=0, out=out, mode="clip")
            return out, self.labels[idx]
        return self.images[idx], self.labels[idx]


def tiny_net(num_classes: int) -> nn.Module:
    """A few-hundred-parameter CNN, fast enough to train in a unit test."""
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, num_classes))


def tiny_registry() -> ModelRegistry:
    """A registry whose only model is tiny_net, registered as 'tiny'."""
    reg = ModelRegistry()
    reg.register_external("tiny", tiny_net, repo="tests", commit="none")
    return reg
