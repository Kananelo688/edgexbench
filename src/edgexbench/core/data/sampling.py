"""
FILENAME:       src/edgexbench/core/data/sampling.py

DESCRIPTION:    Index selection and batching for evaluation and calibration subsets. Operates on
                counts and group keys only; it has no knowledge of images, files or tasks.
                Centralised because dataset caches are written in class-sorted order, which makes
                a prefix slice a biased sample rather than a representative one.
AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Iterator, Sequence

import numpy as np

# Indices are returned as int64 throughout. Fixing the dtype keeps the provenance digest stable
# across platforms, where the default integer width differs between Linux and Windows.
INDEX_DTYPE = np.int64

# Selection strategies recorded in SubsetSpec.kind.
KIND_FULL = "full"
KIND_RANDOM = "random"
KIND_STRATIFIED = "stratified"


class SamplingError(ValueError):
    """Raised when a subset request is inconsistent with the population it describes."""


def _rng(seed: int) -> np.random.Generator:
    """
    Construct the generator used for every selection in this module.

    A single construction point guarantees that two callers asking for the same subset with the
    same seed receive identical indices, which is what makes a PyTorch result and a Hailo result
    comparable on the same images.

    Args:
        seed: Seed value.

    Returns:
        A seeded NumPy generator.
    """
    return np.random.default_rng(seed)


def indices_digest(indices: np.ndarray) -> str:
    """
    Compute a short, stable digest of a selected index set.

    Recorded alongside results so a reported metric can be traced to the exact images that
    produced it. Without it, a subset accuracy is not reproducible.

    Args:
        indices: Selected indices.

    Returns:
        The first 16 hex characters of the SHA-256 of the indices in canonical int64 form.
    """
    return hashlib.sha256(
        np.ascontiguousarray(indices, dtype=INDEX_DTYPE).tobytes()
    ).hexdigest()[:16]


@dataclass(frozen=True)
class SubsetSpec:
    """
    Provenance record describing how a subset was selected.

    Attributes:
        kind: Selection strategy; one of KIND_FULL, KIND_RANDOM, KIND_STRATIFIED.
        n_selected: Number of items selected.
        n_total: Size of the population selected from.
        seed: Seed used, or None when the full population was taken.
        digest: Digest of the selected indices, or None for the full population.
        n_groups: Number of distinct groups present in the selection, stratified only.
    """

    kind: str
    n_selected: int
    n_total: int
    seed: int | None = None
    digest: str | None = None
    n_groups: int | None = None

    @classmethod
    def full(cls, n_total: int) -> "SubsetSpec":
        """
        Build a spec representing the whole population.

        Args:
            n_total: Population size.

        Returns:
            A spec with kind KIND_FULL and no seed or digest, since neither applies.
        """
        return cls(kind=KIND_FULL, n_selected=int(n_total), n_total=int(n_total))

    @classmethod
    def from_indices(cls, kind: str, indices: np.ndarray, n_total: int,
                     seed: int, n_groups: int | None = None) -> "SubsetSpec":
        """
        Build a spec from a realised index set.

        Args:
            kind: Selection strategy used.
            indices: The selected indices.
            n_total: Population size.
            seed: Seed used to produce the selection.
            n_groups: Number of distinct groups represented, stratified only.

        Returns:
            A populated spec including the index digest.
        """
        return cls(kind=kind, n_selected=int(len(indices)), n_total=int(n_total),
                   seed=int(seed), digest=indices_digest(indices), n_groups=n_groups)

    def as_dict(self, prefix: str = "subset_") -> dict:
        """
        Flatten the spec for inclusion in a results row.

        Args:
            prefix: String prepended to each field name, to avoid collisions in a wide CSV.

        Returns:
            Mapping of prefixed field name to value.
        """
        return {f"{prefix}{k}": v for k, v in asdict(self).items()}


def random_subset(n_total: int, n_want: int, seed: int = 0) -> np.ndarray | None:
    """Select a uniformly random subset of item indices.

    The default strategy for evaluation subsets: a uniform sample is unbiased and its sampling
    error is straightforward to state. Prefer stratified_subset for calibration, where a few
    hundred images must still cover every class.

    Returned indices are sorted ascending. Order within an evaluation is irrelevant, and
    sequential access to a memory-mapped cache is substantially faster than random access.

    Args:
        n_total: Population size.
        n_want: Number of items wanted. Values of 0 or None, or any value at or above n_total,
            select the whole population.
        seed: Seed, fixed so every backend scores the same images.

    Returns:
        Sorted int64 indices, or None to signal the whole population. None rather than
        arange(n_total) so callers have one unambiguous "no subsetting" branch.

    Raises:
        SamplingError: If n_total is not positive, or n_want is negative.
    """
    if n_total <= 0:
        raise SamplingError(f"n_total must be positive, got {n_total}")
    if n_want is None or n_want == 0:
        return None
    if n_want < 0:
        raise SamplingError(f"n_want must be non-negative, got {n_want}")
    if n_want >= n_total:
        return None
    idx = _rng(seed).choice(n_total, size=n_want, replace=False)
    return np.sort(idx).astype(INDEX_DTYPE)


def stratified_subset(groups: Sequence | np.ndarray, n_want: int,
                      seed: int = 0) -> np.ndarray | None:
    """
    Select a subset spread as evenly as possible across groups.

    Used where the subset is small relative to the number of groups and every group must still
    be represented, which is the case for INT8 calibration: 256 images across 100 classes.
    A uniform random sample of that size would leave some classes absent and bias the
    activation ranges the quantiser derives.

    Groups holding fewer items than their quota contribute everything they have, and the
    shortfall is redistributed over the remaining groups, so the returned count equals n_want
    whenever the population is large enough.

    The remainder, when the quota does not divide evenly, is assigned to randomly chosen groups
    rather than to the first few. Assigning by position would systematically over-represent
    whichever classes sort earliest.

    Args:
        groups: Group key per item, length n_total. Keys may be any hashable value, so a task
            may stratify on class label, object count, or a size bucket.
        n_want: Number of items wanted. Values of 0 or None, or any value at or above the
            population size, select the whole population.
        seed: Seed, fixed so the selection is reproducible.

    Returns:
        Sorted int64 indices, or None to signal the whole population.

    Raises:
        SamplingError: If groups is empty or n_want is negative.
    """
    keys = np.asarray(groups)
    n_total = len(keys)
    if n_total == 0:
        raise SamplingError("groups is empty")
    if n_want is None or n_want == 0:
        return None
    if n_want < 0:
        raise SamplingError(f"n_want must be non-negative, got {n_want}")
    if n_want >= n_total:
        return None

    rng = _rng(seed)

    # Member indices per group, in a deterministic group order. np.unique sorts the keys, so the
    # grouping does not depend on the order items happen to appear in.
    uniq = np.unique(keys)
    members = {g: np.flatnonzero(keys == g) for g in uniq}

    taken: dict = {g: 0 for g in uniq}
    active = [g for g in uniq if len(members[g]) > 0]
    remaining = n_want

    # Repeated passes rather than a single division: groups smaller than the quota exhaust early,
    # and their unused allocation must be redistributed over the groups that still have items.
    while remaining > 0 and active:
        base, extra = divmod(remaining, len(active))
        bonus = set(rng.choice(len(active), size=extra, replace=False).tolist()) if extra else set()
        progressed = False
        for i, g in enumerate(list(active)):
            want = base + (1 if i in bonus else 0)
            can = min(want, len(members[g]) - taken[g])
            if can > 0:
                taken[g] += can
                remaining -= can
                progressed = True
            if taken[g] >= len(members[g]):
                active.remove(g)
        # Guard against a non-terminating loop if every active group is already exhausted.
        if not progressed:
            break

    picked: list[np.ndarray] = []
    for g in uniq:
        k = taken[g]
        if k:
            # Which items within the group is also random: taking the first k would bias towards
            # whatever ordering the cache was written in.
            sel = rng.choice(len(members[g]), size=k, replace=False)
            picked.append(members[g][sel])

    idx = np.concatenate(picked) if picked else np.empty(0, INDEX_DTYPE)
    return np.sort(idx).astype(INDEX_DTYPE)


def iter_batches(n_total: int, batch_size: int,
                 indices: np.ndarray | None = None) -> Iterator[np.ndarray]:
    """
    Yield index arrays covering a population or a selected subset.

    Yields indices rather than data so this module stays independent of how items are stored,
    which matters once detection introduces a container other than a dense array.

    Args:
        n_total: Population size, used when indices is None.
        batch_size: Maximum items per batch. The final batch is shorter when the count does not
            divide evenly.
        indices: Selected indices to iterate over. None iterates the whole population in order.

    Yields:
        int64 index arrays. The concatenation of all yielded arrays is exactly the selection,
        with no item repeated or omitted.

    Raises:
        SamplingError: If batch_size is not positive, or n_total is not positive when no
            indices are supplied.
    """
    if batch_size <= 0:
        raise SamplingError(f"batch_size must be positive, got {batch_size}")

    if indices is None:
        if n_total <= 0:
            raise SamplingError(f"n_total must be positive, got {n_total}")
        source = np.arange(n_total, dtype=INDEX_DTYPE)
    else:
        source = np.ascontiguousarray(indices, dtype=INDEX_DTYPE)

    n = len(source)
    for start in range(0, n, batch_size):
        # [AUDIT S1] The old comment claimed the slice would overrun without min(); numpy
        # slices already stop at the end of the array. The clamp is kept only to make the bound
        # explicit.
        # Before:
        #   # min(start + batch_size, n) rather than start + batch_size: without the clamp the final
        #   # slice overruns the selection and the caller silently processes more items than it
        #   # requested, which corrupts any metric computed over a subset.
        yield source[start:min(start + batch_size, n)]


def count_batches(n_items: int, batch_size: int) -> int:
    """
    Number of batches iter_batches will yield for a given count.

    Provided so callers can size a progress bar without materialising the iterator.

    Args:
        n_items: Number of items to be iterated.
        batch_size: Maximum items per batch.

    Returns:
        Number of batches, zero when n_items is zero.

    Raises:
        SamplingError: If batch_size is not positive.
    """
    if batch_size <= 0:
        raise SamplingError(f"batch_size must be positive, got {batch_size}")
    return (int(n_items) + batch_size - 1) // batch_size


def selection_size(n_total: int, indices: np.ndarray | None) -> int:
    """Resolve how many items a selection covers.

    Args:
        n_total: Population size.
        indices: Selected indices, or None for the whole population.

    Returns:
        The number of items the selection covers.
    """
    return int(n_total) if indices is None else int(len(indices))