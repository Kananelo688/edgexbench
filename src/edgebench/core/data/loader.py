"""
FILENAME:       src/edgebench/core/data/loader.py

DESCRIPTION:   This is from-scratch, custom loader module. It implements custom batch iteration over a decoded image cache, 
                and device-side augmentation. Replaces torch.utils.data.DataLoader: with images already decoded, a batch is a
                single indexed read from a memory map, so worker processes add overhead and
                failure modes without adding throughput. Augmentation runs on the accelerator
                for the same reason - performing it per-image on the host would reinstate the
                CPU bottleneck the cache exists to remove.

AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import functools
import math
import queue
import threading
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torchvision.ops import roi_align

from edgebench.core.data.cache import ImageCache
from edgebench.core.data.sampling import iter_batches

# ImageNet-1k channel statistics, the torchvision and timm defaults. Held in [0,1] space and
# scaled at use, since cached pixels are raw uint8.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class LoaderError(RuntimeError):
    """Raised when a loader is configured inconsistently with the cache it reads."""


@dataclass(frozen=True)
class Normalisation:
    """
    Channel normalisation applied after augmentation.

    Attributes:
        mean: Per-channel means in [0,1] space.
        std: Per-channel standard deviations in [0,1] space.
    """
    #These must be replaced with statistics for different datasets.
    # These statistics can be replaced by those of a different datasets.
    mean: tuple[float, float, float] = IMAGENET_MEAN 
    std: tuple[float, float, float] = IMAGENET_STD 

    def tensors(self, device: torch.device | str,
                dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise mean and standard deviation as broadcastable NCHW tensors.

        Scaled by 255 so they apply directly to raw pixel values, avoiding a separate division
        pass over the batch.

        Args:
            device: Device to place the tensors on.
            dtype: Element type.

        Returns:
            Mean and standard deviation, each shaped (1, 3, 1, 1) and scaled to [0,255].
        """
        # [AUDIT L4] Built on every call, i.e. two small synchronous host-to-device copies per
        # batch. Now cached per (statistics, device, dtype). The cached tensors are shared, so
        # callers must not modify them in place.
        # Before:
        #   m = torch.tensor(self.mean, device=device, dtype=dtype).view(1, 3, 1, 1) * 255.0
        #   s = torch.tensor(self.std, device=device, dtype=dtype).view(1, 3, 1, 1) * 255.0
        #   return m, s
        return _norm_tensors(self.mean, self.std, torch.device(device), dtype)


@functools.lru_cache(maxsize=None)
def _norm_tensors(mean: tuple[float, float, float], std: tuple[float, float, float],
                  device: torch.device,
                  dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the scaled normalisation tensors once per combination.

    Args:
        mean: Per-channel means in [0,1] space.
        std: Per-channel standard deviations in [0,1] space.
        device: Device to place the tensors on.
        dtype: Element type.

    Returns:
        Mean and standard deviation, each shaped (1, 3, 1, 1) and scaled to [0,255].
    """
    m = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1) * 255.0
    s = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1) * 255.0
    return m, s


def normalise(x: torch.Tensor, norm: Normalisation) -> torch.Tensor:
    """Apply channel normalisation to a float NCHW batch in [0,255].

    Args:
        x: Float NCHW batch with raw pixel magnitudes.
        norm: Normalisation constants.

    Returns:
        The normalised batch.
    """
    mean, std = norm.tensors(x.device, x.dtype)
    return (x - mean) / std


def to_nchw_float(batch_u8: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 NHWC batch to float NCHW without normalising.

    Args:
        batch_u8: uint8 NHWC batch.

    Returns:
        Float NCHW batch with values still in [0,255].
    """
    return batch_u8.permute(0, 3, 1, 2).float()


def random_resized_crop_flip(batch_u8: torch.Tensor,
                             out_res: int,
                             scale: tuple[float, float],
                             ratio: tuple[float, float],
                             hflip_prob: float,
                             generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Apply per-sample random-resized-crop and horizontal flip to a whole batch on device.

    Crops are extracted with roi_align, which cuts and bilinearly resamples every box in one
    batched call; a per-image loop would cost more than the convolutions that follow.

    NOTE: this is not bit-identical to torchvision.transforms.RandomResizedCrop. That
    implementation retries up to ten times to fit a box inside the image and falls back to a
    centre crop, whereas this one draws once and clamps the box to the image bounds. A clamped
    box no longer has the drawn aspect ratio, so it is stretched when resampled to a square;
    this is frequent for scale draws near 1.0 with a non-square ratio. The same procedure is
    applied to every architecture, so it cannot bias a comparison between them. (Audit L7:
    documented, not changed.)

    Args:
        batch_u8: uint8 NHWC batch of square images.
        out_res: Output side length.
        scale: Crop area range as a fraction of the source image.
        ratio: Crop aspect-ratio range.
        hflip_prob: Horizontal flip probability.
        generator: Optional generator for reproducible augmentation.

    Returns:
        Float NCHW batch at out_res, values in [0,255], not yet normalised.

    Raises:
        LoaderError: If the batch is not square uint8 NHWC.
    """
    # [AUDIT L6] The message promised uint8 but the dtype was never checked.
    # Before:
    #   if batch_u8.ndim != 4 or batch_u8.shape[3] != 3 or batch_u8.shape[1] != batch_u8.shape[2]:
    #       raise LoaderError(f"expected square uint8 NHWC, got {tuple(batch_u8.shape)}")
    if batch_u8.dtype != torch.uint8 or batch_u8.ndim != 4 or batch_u8.shape[3] != 3 \
            or batch_u8.shape[1] != batch_u8.shape[2]:
        raise LoaderError(f"expected square uint8 NHWC, got {batch_u8.dtype} "
                          f"{tuple(batch_u8.shape)}")

    n, src = batch_u8.shape[0], batch_u8.shape[1]
    x = to_nchw_float(batch_u8)
    dev = x.device

    def rand(size: int) -> torch.Tensor:
        return torch.rand(size, device=dev, generator=generator)

    area = src * src * (scale[0] + (scale[1] - scale[0]) * rand(n))
    # Log-uniform aspect ratio keeps portrait and landscape crops equally likely; sampling the
    # ratio uniformly would favour landscape.
    log_lo, log_hi = math.log(ratio[0]), math.log(ratio[1])
    ar = torch.exp(log_lo + (log_hi - log_lo) * rand(n))

    w = torch.sqrt(area * ar).clamp(max=float(src))
    h = torch.sqrt(area / ar).clamp(max=float(src))
    x0 = rand(n) * (src - w)
    y0 = rand(n) * (src - h)

    boxes = torch.stack(
        [torch.arange(n, device=dev, dtype=x.dtype), x0, y0, x0 + w, y0 + h], dim=1)
    # sampling_ratio=2 averages a 2x2 grid per output pixel, giving mild anti-aliasing. Crops
    # here are mostly upscaled, so heavier filtering would cost time for no benefit.
    out = roi_align(x, boxes, (out_res, out_res), spatial_scale=1.0,
                    sampling_ratio=2, aligned=True)

    if hflip_prob > 0:
        flip = rand(n) < hflip_prob
        out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out


def centre_view(batch_u8: torch.Tensor) -> torch.Tensor:
    """Convert an evaluation batch to float NCHW with no augmentation.

    Evaluation images are already resized and centre-cropped at cache-build time, so no
    geometric transform is applied here.

    Args:
        batch_u8: uint8 NHWC batch.

    Returns:
        Float NCHW batch in [0,255].
    """
    return to_nchw_float(batch_u8)


class CacheLoader:
    """Iterates batches from an ImageCache, prefetching on a background thread.

    Yields pinned uint8 NHWC tensors and int64 labels. Preprocessing is deliberately left to
    the caller so that the augmented training path and the plain evaluation path share one
    source of batches.
    """

    def __init__(self,
                 cache: ImageCache,
                 batch_size: int,
                 *,
                 shuffle: bool,
                 drop_last: bool | None = None,
                 seed: int = 0,
                 indices: np.ndarray | None = None,
                 limit_batches: int | None = None,
                 prefetch: int = 4,
                 pin_memory: bool = True) -> None:
        """Initialise the loader.

        Args:
            cache: Cache to read from.
            batch_size: Items per batch.
            shuffle: Whether to reshuffle each epoch. Training shuffles; evaluation does not.
            drop_last: Discard a short final batch. Defaults to the value of `shuffle`, because
                training needs a constant batch size for mixup and stable normalisation
                statistics, while evaluation must score every item.
            seed: Base seed. The per-epoch permutation is seeded with seed + epoch, so the
                order is random across epochs yet reproducible and resume-safe.
            indices: Restrict iteration to this subset of the cache.
            limit_batches: Cap batches per epoch, for smoke tests.
            prefetch: Batches prepared ahead of the consumer.
            pin_memory: Allocate batches in pinned memory, enabling asynchronous host-to-device
                copies.

        Raises:
            LoaderError: If batch_size or prefetch is not positive, indices are out of range,
                or the configuration yields no batches.
        """
        if batch_size <= 0:
            raise LoaderError(f"batch_size must be positive, got {batch_size}")
        if prefetch < 1:
            raise LoaderError(f"prefetch must be at least 1, got {prefetch}")

        self.cache = cache
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = shuffle if drop_last is None else drop_last
        self.seed = seed
        self.prefetch = prefetch
        self.pin_memory = pin_memory
        self.epoch = 0

        n_cache = len(cache)
        if indices is None:
            self.indices = None
            self.n_items = n_cache
        else:
            idx = np.asarray(indices)
            if idx.size and (idx.min() < 0 or idx.max() >= n_cache):
                raise LoaderError(
                    f"indices span [{idx.min()}, {idx.max()}], cache holds {n_cache}")
            self.indices = idx
            self.n_items = len(idx)

        full = self.n_items // batch_size if self.drop_last \
            else math.ceil(self.n_items / batch_size)
        self.n_batches = min(full, limit_batches) if limit_batches else full
        # [AUDIT L5] drop_last with fewer items than one batch gave zero batches, so a training
        # epoch silently did nothing.
        # Before: (no check)
        if self.n_batches == 0:
            raise LoaderError(
                f"no batches: {self.n_items} items, batch_size {batch_size}, "
                f"drop_last={self.drop_last}")

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation used for the next iteration.

        Args:
            epoch: Epoch number.

        Returns:
            None.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Number of batches per epoch.

        Returns:
            The batch count, after drop_last and limit_batches are applied.
        """
        return self.n_batches

    @property
    def items_per_epoch(self) -> int:
        """Number of items actually visited per epoch.

        Returns:
            Batch count multiplied by batch size, except for a retained short final batch.
        """
        if self.drop_last:
            return self.n_batches * self.batch_size
        return min(self.n_items, self.n_batches * self.batch_size)

    def _order(self) -> np.ndarray:
        """Resolve the item order for the current epoch.

        Returns:
            Indices into the cache, shuffled when shuffle is enabled.
        """
        base = np.arange(len(self.cache)) if self.indices is None else self.indices
        if not self.shuffle:
            return base
        # [AUDIT L2] seed + epoch made seed 42 / epoch 1 identical to seed 43 / epoch 0,
        # correlating the data order of runs meant to estimate seed noise. Seeding with the
        # pair goes through a SeedSequence, giving independent streams.
        # Before:
        #   return np.random.default_rng(self.seed + self.epoch).permutation(base)
        return np.random.default_rng([self.seed, self.epoch]).permutation(base)

    def _put(self, out: queue.Queue, item, stop: threading.Event) -> bool:
        """Place an item on the queue unless the consumer has gone away.

        Args:
            out: Bounded queue shared with the consumer.
            item: Item to enqueue.
            stop: Set by the consumer when it stops iterating.

        Returns:
            True if the item was enqueued, False if the consumer stopped first.
        """
        while not stop.is_set():
            try:
                out.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _read(self, batch_idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one batch from the cache as tensors.

        Args:
            batch_idx: Cache indices, sorted.

        Returns:
            uint8 NHWC images and int64 labels, pinned when pin_memory is set.
        """
        # [AUDIT L3] Every batch used to be copied three times: by fancy indexing, by an
        # np.array in ImageCache.take, and by pin_memory(). With pinning, images are now read
        # straight into a freshly allocated pinned tensor: one copy. Allocating per batch
        # rather than reusing a ring is deliberate: torch's caching host allocator will not
        # hand a block back out while a non_blocking copy from it is still in flight, which a
        # hand-rolled ring could not guarantee.
        # Before:
        #   images, labels = self.cache.take(np.sort(batch_idx))
        #   xb = torch.from_numpy(images)
        #   yb = torch.from_numpy(labels)
        #   if self.pin_memory:
        #       xb, yb = xb.pin_memory(), yb.pin_memory()
        if self.pin_memory:
            res = self.cache.resolution
            xb = torch.empty((len(batch_idx), res, res, 3), dtype=torch.uint8, pin_memory=True)
            _, labels = self.cache.take(batch_idx, out=xb.numpy())
            return xb, torch.from_numpy(labels).pin_memory()
        images, labels = self.cache.take(batch_idx)
        return torch.from_numpy(images), torch.from_numpy(labels)

    def _produce(self, order: np.ndarray, out: queue.Queue, stop: threading.Event) -> None:
        """Read batches on a background thread and place them on the queue.

        Args:
            order: Item order for this epoch.
            out: Bounded queue receiving (images, labels) or a sentinel.
            stop: Set by the consumer when it stops iterating; the thread then exits.

        Returns:
            None.
        """
        try:
            for k, batch_idx in enumerate(iter_batches(len(order), self.batch_size, order)):
                if k >= self.n_batches:
                    break
                # Sorting within the batch converts scattered memory-map reads into a mostly
                # sequential pattern; order inside a batch has no effect on training.
                if not self._put(out, self._read(np.sort(batch_idx)), stop):
                    return
            self._put(out, None, stop)
        except Exception as exc:  # re-raised in the consumer so errors are not swallowed
            self._put(out, exc, stop)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate one epoch.

        Yields:
            Pinned uint8 NHWC image tensors and int64 label tensors.

        Raises:
            Exception: Any exception raised by the producer thread, re-raised here.
        """
        # [AUDIT L1] If the consumer stopped early (an exception mid-epoch, or a break), the
        # producer blocked forever on a full queue, leaking a thread and its pinned batches for
        # every failed model in a sweep. The consumer now signals the producer to exit and
        # drains the queue so a blocked put returns.
        # Before:
        #   threading.Thread(target=self._produce, args=(order, q), daemon=True).start()
        #   while True:
        #       item = q.get()
        #       ...
        #       yield item
        order = self._order()
        q: queue.Queue = queue.Queue(self.prefetch)
        stop = threading.Event()
        # Daemon thread: a crash in the consumer must not leave the process unable to exit.
        producer = threading.Thread(target=self._produce, args=(order, q, stop), daemon=True)
        producer.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            producer.join(timeout=5.0)