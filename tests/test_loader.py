"""CacheLoader: coverage, seeding and thread lifecycle (audit L1, L2, L5)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from edgexbench.core.data.loader import CacheLoader, LoaderError

from helpers import FakeCache


def labels_seen(loader) -> np.ndarray:
    return np.concatenate([yb.numpy() for _, yb in loader])


def test_each_item_exactly_once_per_epoch():
    cache = FakeCache(103, 4)
    cache.labels = np.arange(103, dtype=np.int64)        # label == index, so it identifies items
    loader = CacheLoader(cache, 10, shuffle=True, drop_last=False, pin_memory=False)
    for epoch in range(3):
        loader.set_epoch(epoch)
        assert sorted(labels_seen(loader).tolist()) == list(range(103))


def test_seed_streams_do_not_overlap_across_epochs():
    cache = FakeCache(50, 4)
    a = CacheLoader(cache, 10, shuffle=True, seed=42, pin_memory=False)
    b = CacheLoader(cache, 10, shuffle=True, seed=43, pin_memory=False)
    a.set_epoch(1)
    b.set_epoch(0)
    assert not np.array_equal(a._order(), b._order())


def test_abandoned_epoch_does_not_leak_threads():
    loader = CacheLoader(FakeCache(1000, 4), 10, shuffle=True, prefetch=2, pin_memory=False)
    before = threading.active_count()
    for _ in range(5):
        for i, _batch in enumerate(loader):
            if i == 0:
                break
    time.sleep(0.5)
    assert threading.active_count() <= before


def test_zero_batch_loader_rejected():
    with pytest.raises(LoaderError):
        CacheLoader(FakeCache(5, 4), 10, shuffle=True, pin_memory=False)
