"""
FILENAME:       src/edgebench/core/data/cache.py

DESCRIPTION:    This builds a decoded image cache for cases where training compute is not sufficient.
                Since decoding images during training (which happens on a CPU) can be slow, it can stall the GPU and result in
                inefficient training speed. This module define routines that writes a dataset and decode it once into a uint8 NHWC memory-mapped array
                plus labels and provenance, and reads it back. 
                
                The storage format is task-agnostic. The preprocessing transform is supplied by
                the caller, so classification resize-and-crop and detection letterboxing use the same machinery.

AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# Extensions treated as images during discovery. Compared lowercased, because ImageNet ships
# uppercase .JPEG and a case-sensitive match would silently find nothing on Linux.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)

# Directory names never treated as a class during discovery. Guards against building a training
# cache that silently includes the evaluation split.
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {"val", "val.x", "valid", "validation", "test", "testing"}
)

# Labels are int64 throughout: it is torch's index dtype, so no conversion is needed at use.
LABEL_DTYPE = np.int64

# Images decoded per thread-pool submission. Bounds peak memory during the build and sets the
# progress-reporting granularity.
BUILD_CHUNK: int = 512

# How a sample's label is read from its path. The two conventions conflict, so the caller must
# choose; neither can be inferred safely from the tree.
#   "parent": the directory containing the image (Kaggle ImageNet-100: <part>/<class>/x.JPEG).
#   "top":    the first directory below the root (Tiny-ImageNet: <class>/images/x.JPEG).
LABEL_FROM: tuple[str, ...] = ("parent", "top")


class CacheError(RuntimeError):
    """Raised when a cache cannot be built, is malformed, or is inconsistent with its request."""


# A transform maps a decoded PIL image to an HWC uint8 array of the cache's fixed size.
Transform = Callable[[Any], np.ndarray]



@dataclass(frozen=True)
class DiscoveredSamples:
    """Result of walking a class-folder dataset tree.

    Attributes:
        samples: (file path, class index) pairs, sorted by path.
        classes: Class names, sorted, indexed by class index.
        roots: Directories that were walked.
    """

    samples: list[tuple[Path, int]]
    classes: list[str]
    roots: list[Path]

    def __len__(self) -> int:
        """Number of discovered samples.

        Returns:
            The sample count.
        """
        return len(self.samples)

    def counts_per_class(self) -> np.ndarray:
        """Count samples in each class.

        Returns:
            Array of length len(classes) giving the sample count per class index.
        """
        return np.bincount(np.array([c for _, c in self.samples], dtype=LABEL_DTYPE),
                           minlength=len(self.classes))


def discover_class_folders(roots: Sequence["str | Path"],
                           classes: Sequence[str] | None = None,
                           exclude_dirs: Iterable[str] = EXCLUDED_DIR_NAMES,
                           extensions: Iterable[str] = IMAGE_EXTENSIONS,
                           *,
                           label_from: str = "parent") -> DiscoveredSamples:
    """
    Walk one or more roots holding per-class sub-directories.

    With label_from="parent" (the default), a sample's label is the name of the directory
    immediately containing it, so the same class appearing under several roots, or under
    several part directories of one root, is merged. This is required by datasets distributed
    in parts, such as the Kaggle ImageNet-100 release, where the four train parts each hold a
    different subset of classes. With label_from="top" the label is the first directory below
    the root, for trees that nest images inside each class (Tiny-ImageNet's <class>/images/).

    Results are sorted by root, then by relative POSIX path, so the sample order and therefore
    the label digest are identical on every run and every operating system.

    Args:
        roots: Directories to walk.
        classes: Canonical class list to validate against. None infers the list from what is
            found. Pass the training split's list when building a validation cache, so a class
            index means the same thing in both.
        exclude_dirs: Directory names to skip, compared case-insensitively against every path
            component.
        extensions: Accepted file extensions, compared lowercased.
        label_from: One of LABEL_FROM: "parent" or "top".

    Returns:
        The discovered samples and class list.

    Raises:
        CacheError: If a root is missing, label_from is unknown, no images are found, the
            canonical list holds duplicates, or the discovered class set differs from it.
    """
    if label_from not in LABEL_FROM:
        raise CacheError(f"label_from must be one of {LABEL_FROM}, got {label_from!r}")
    roots = [Path(r).expanduser().resolve() for r in roots]
    for r in roots:
        if not r.is_dir():
            raise CacheError(f"not a directory: {r}")

    exts = {e.lower() for e in extensions}
    skip = {d.lower() for d in exclude_dirs}

    found: dict[str, list[tuple[int, str, Path]]] = {}
    for root_index, root in enumerate(roots):
        for p in root.rglob("*"):
            if p.suffix.lower() not in exts or not p.is_file():
                continue
            # [AUDIT C7] Hidden files such as macOS "._x.jpg" resource forks carry an image
            # extension but are not images, and aborted the build at decode time.
            # Before: (no check)
            if p.name.startswith("."):
                continue
            rel = p.parent.relative_to(root)
            if {part.lower() for part in rel.parts} & skip:
                continue
            # [AUDIT C1] A file directly in a root used to be labelled with the root's own name,
            # contradicting the comment that the label is never the part or root name. Nested
            # layouts are handled by label_from="top" rather than by changing the default, which
            # would collapse the Kaggle layout to its part directories when walked from the
            # dataset root.
            # Before:
            #   # The label is the immediately containing directory, never the part or root name.
            #   label = p.parent.name
            if not rel.parts:
                continue
            label = p.parent.name if label_from == "parent" else rel.parts[0]
            # [AUDIT C6] Sorting by absolute Path is case-insensitive on Windows and
            # case-sensitive on POSIX, so the order differed between them. The sort key is now
            # (root index, relative POSIX path).
            # Before: found.setdefault(label, []).append(p)
            found.setdefault(label, []).append(
                (root_index, p.relative_to(root).as_posix(), p))

    if not found:
        raise CacheError(f"no images found under {[str(r) for r in roots]}")

    discovered = sorted(found)
    if classes is None:
        class_list = discovered
    else:
        class_list = list(classes)
        # [AUDIT C7] Duplicates passed the set comparison below but left the index map wrong.
        # Before: (no check)
        if len(set(class_list)) != len(class_list):
            dupes = sorted({c for c in class_list if class_list.count(c) > 1})
            raise CacheError(f"canonical class list has duplicate entries: {dupes[:5]}")
        if set(discovered) != set(class_list):
            missing = sorted(set(class_list) - set(discovered))
            extra = sorted(set(discovered) - set(class_list))
            raise CacheError(
                f"class mismatch against the canonical list: "
                f"{len(missing)} missing {missing[:5]}, {len(extra)} unexpected {extra[:5]}")

    index = {c: i for i, c in enumerate(class_list)}
    # Before:
    #   samples = sorted(((p, index[label]) for label, paths in found.items() for p in paths),
    #                    key=lambda t: t[0])
    keyed = sorted((root_index, rel, p, index[label])
                   for label, entries in found.items() for root_index, rel, p in entries)
    samples = [(p, label_index) for _, _, p, label_index in keyed]
    return DiscoveredSamples(samples=samples, classes=class_list, roots=roots)



@dataclass
class CacheMetadata:
    """Provenance sidecar written beside every cache.

    Attributes:
        dataset: Dataset identifier.
        split: Split identifier, for example "train" or "val".
        resolution: Square side length of the cached images.
        n_samples: Number of cached images.
        n_classes: Number of classes.
        classes: Class names, indexed by label value.
        layout: Array layout, always "NHWC".
        dtype: Array element type, always "uint8".
        value_range: Description of the stored pixel range.
        transform: Human-readable description of the preprocessing applied before caching.
        source_roots: Directories the images were read from.
        label_digest: Digest of the label array, used to detect a cache rebuilt from a changed
            dataset.
        built_at: Build timestamp.
        build_seconds: Wall-clock build duration.
    """

    dataset: str
    split: str
    resolution: int
    n_samples: int
    n_classes: int
    classes: list[str]
    layout: str = "NHWC"
    dtype: str = "uint8"
    value_range: str = "[0,255] raw pixels, not normalised"
    transform: str = ""
    source_roots: list[str] = field(default_factory=list)
    label_digest: str = ""
    built_at: str = ""
    build_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise for writing to JSON.

        Returns:
            Mapping of field name to value.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetadata":
        """Construct from a mapping, ignoring fields written by a newer version.

        Unknown keys are tolerated here, unlike in configuration loading, because a cache
        written by a later release must remain readable rather than aborting a run.

        Args:
            data: Mapping of field name to value.

        Returns:
            The constructed metadata.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def _digest(array: np.ndarray) -> str:
    """Compute a short digest of an array's bytes.

    Args:
        array: Array to digest.

    Returns:
        The first 16 hex characters of the SHA-256 of the contiguous buffer.
    """
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]



def cache_paths(cache_dir: "str | Path", split: str, resolution: int) -> dict[str, Path]:
    """Resolve the three files making up one cache.

    The resolution is part of the filename so that changing it builds a new cache rather than
    silently reusing one at the wrong size.

    Args:
        cache_dir: Directory holding the dataset's caches.
        split: Split identifier.
        resolution: Square side length.

    Returns:
        Mapping with keys "images", "labels" and "meta".
    """
    d = Path(cache_dir)
    stem = f"{split}_{resolution}"
    return {"images": d / f"{stem}_x.npy",
            "labels": d / f"{stem}_y.npy",
            "meta": d / f"{stem}.json"}


def cache_exists(cache_dir: "str | Path", split: str, resolution: int) -> bool:
    """Report whether a complete cache is present.

    All three files must exist. A partial set indicates an interrupted build and is treated as
    absent, so the next run rebuilds rather than reading truncated data.

    Args:
        cache_dir: Directory holding the dataset's caches.
        split: Split identifier.
        resolution: Square side length.

    Returns:
        True when images, labels and metadata are all present.
    """
    return all(p.exists() for p in cache_paths(cache_dir, split, resolution).values())



def build_cache(samples: Sequence[tuple[Path, int]],
                transform: Transform,
                cache_dir: "str | Path",
                split: str,
                resolution: int,
                *,
                dataset: str,
                classes: Sequence[str],
                transform_description: str = "",
                source_roots: Sequence["str | Path"] = (),
                workers: int | None = None,
                overwrite: bool = False,
                progress: bool = True) -> CacheMetadata:
    """Decode a sample list into a uint8 NHWC cache.

    Decoding runs on a thread pool rather than worker processes. PIL releases the GIL while
    decoding, so threads reach comparable throughput, and they avoid the process-spawn failure
    seen with DataLoader workers under Jupyter on Windows.

    The image array is written through a memory map so a dataset larger than RAM can be built,
    and is published by renaming a temporary file, so an interrupted build can never be mistaken
    for a complete cache.

    Args:
        samples: (file path, class index) pairs, in the order they will be stored.
        transform: Callable mapping a decoded PIL image to an HWC uint8 array of side
            `resolution`.
        cache_dir: Directory to write into; created if absent.
        split: Split identifier.
        resolution: Square side length the transform produces.
        dataset: Dataset identifier, recorded in metadata.
        classes: Class names indexed by label value, recorded in metadata.
        transform_description: Human-readable description of the transform, recorded in
            metadata so the preprocessing is recoverable without reading code.
        source_roots: Directories the samples came from, recorded in metadata.
        workers: Decoding threads. None uses twice the CPU count, capped at 12, since the work
            is part I/O.
        overwrite: Rebuild even when a complete cache is present.
        progress: Emit progress lines during the build.

    Returns:
        The metadata written alongside the cache.

    Raises:
        CacheError: If the cache exists and overwrite is False, the sample list is empty, a
            label is out of range, or the transform returns an array of the wrong shape or type.
    """
    if not samples:
        raise CacheError("sample list is empty")

    paths = cache_paths(cache_dir, split, resolution)
    if cache_exists(cache_dir, split, resolution) and not overwrite:
        raise CacheError(
            f"cache already present for {split}@{resolution}px in {cache_dir}; "
            f"pass overwrite=True to rebuild")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # [AUDIT C2] The old image array used to stay published throughout a rebuild, so a crash
    # after the new labels and metadata were written left a "complete" cache pairing old
    # pixels with new labels, which open() accepted. The previous cache is now withdrawn
    # before anything is written, so an interrupted rebuild reads as absent.
    # Before: (nothing removed; the old files were only replaced at the end)
    for key in ("images", "meta", "labels"):
        paths[key].unlink(missing_ok=True)

    n = len(samples)
    n_classes = len(classes)
    labels = np.empty(n, LABEL_DTYPE)
    for i, (_, lab) in enumerate(samples):
        if not 0 <= lab < n_classes:
            raise CacheError(f"label {lab} at index {i} is outside [0, {n_classes})")
        labels[i] = lab

    if workers is None:
        workers = min(12, (os.cpu_count() or 4) * 2)
    # [AUDIT C8] workers=0 used to surface as a bare ValueError from ThreadPoolExecutor.
    # Before: (no check)
    if workers < 1:
        raise CacheError(f"workers must be at least 1, got {workers}")

    from PIL import Image

    def decode(i: int) -> tuple[int, np.ndarray]:
        """Decode and transform one sample.

        Args:
            i: Index into the sample list.

        Returns:
            The index and its HWC uint8 array.

        Raises:
            CacheError: If the file cannot be decoded or transformed, naming the file.
        """
        path = samples[i][0]
        # [AUDIT C3] A corrupt file raised a bare PIL error with no path, after possibly
        # hours of decoding. The failing file is now named.
        # Before:
        #   with Image.open(path) as im:
        #       arr = transform(im.convert("RGB"))
        #   return i, arr
        try:
            # convert("RGB") is mandatory, not defensive: ImageNet contains greyscale and CMYK
            # files that would otherwise yield one- or four-channel arrays.
            with Image.open(path) as im:
                arr = transform(im.convert("RGB"))
        except Exception as exc:
            raise CacheError(f"failed to decode {path}: {type(exc).__name__}: {exc}") from exc
        return i, arr

    tmp = paths["images"].with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                   shape=(n, resolution, resolution, 3))
    t0 = time.time()
    # [AUDIT C3] A failed build left the multi-gigabyte temporary array behind. It is now
    # removed on any failure, after the memory map is released.
    # Before:
    #   try:
    #       ...
    #   finally:
    #       del mm
    failed = True
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(0, n, BUILD_CHUNK):
                stop = min(start + BUILD_CHUNK, n)
                for i, arr in pool.map(decode, range(start, stop)):
                    if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8 \
                            or arr.shape != (resolution, resolution, 3):
                        raise CacheError(
                            f"transform returned {getattr(arr, 'dtype', type(arr).__name__)} "
                            f"{getattr(arr, 'shape', '')} for {samples[i][0]}, "
                            f"expected uint8 ({resolution}, {resolution}, 3)")
                    mm[i] = arr
                if progress:
                    print(f"  {split}@{resolution}px  {stop}/{n}", end="\r", flush=True)
        mm.flush()
        failed = False
    finally:
        # Releasing the memory map before renaming is required on Windows, where an open
        # handle blocks the replace.
        del mm
        if failed:
            tmp.unlink(missing_ok=True)
    if progress:
        print()

    # [AUDIT C2] Labels and metadata are written through temporary files and renamed, like the
    # image array, so neither can be observed half-written.
    # Before:
    #   np.save(paths["labels"], labels)
    #   ...
    #   paths["meta"].write_text(json.dumps(meta.to_dict(), indent=2))
    labels_tmp = paths["labels"].with_suffix(".tmp.npy")
    np.save(labels_tmp, labels)
    os.replace(labels_tmp, paths["labels"])
    meta = CacheMetadata(
        dataset=dataset, split=split, resolution=resolution, n_samples=n,
        n_classes=n_classes, classes=list(classes),
        transform=transform_description,
        source_roots=[str(r) for r in source_roots],
        label_digest=_digest(labels),
        built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        build_seconds=round(time.time() - t0, 1),
    )
    meta_tmp = paths["meta"].with_suffix(".tmp.json")
    meta_tmp.write_text(json.dumps(meta.to_dict(), indent=2))
    os.replace(meta_tmp, paths["meta"])
    # [AUDIT C8] Comment corrected: cache_exists requires all three files equally. The image
    # array is published last so the set is complete only once labels and metadata are final.
    # Before:
    #   # Published last: the image array is what cache_exists checks for most strongly, so it
    #   # becomes visible only once labels and metadata are already on disk.
    os.replace(tmp, paths["images"])
    return meta



class ImageCache:
    """Read-side handle for one built cache.

    The image array is memory-mapped rather than loaded: the operating system pages in only
    what is touched, and after the first pass the pages are served from the page cache. Access
    in ascending index order is therefore substantially faster than random access, which is why
    callers sort their index selections.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, meta: CacheMetadata) -> None:
        """Initialise from already-loaded components.

        Args:
            images: Memory-mapped uint8 NHWC array.
            labels: int64 label array.
            meta: Cache metadata.
        """
        self._images = images
        self._labels = labels
        self.meta = meta

    @classmethod
    def open(cls, cache_dir: "str | Path", split: str, resolution: int) -> "ImageCache":
        """Open a built cache and validate its internal consistency.

        Args:
            cache_dir: Directory holding the dataset's caches.
            split: Split identifier.
            resolution: Square side length.

        Returns:
            The opened cache.

        Raises:
            CacheError: If any file is missing, or the arrays disagree with each other or with
                the metadata.
        """
        paths = cache_paths(cache_dir, split, resolution)
        missing = [k for k, p in paths.items() if not p.exists()]
        if missing:
            available = sorted(p.name for p in Path(cache_dir).glob("*_x.npy")) \
                if Path(cache_dir).is_dir() else []
            raise CacheError(
                f"cache for {split}@{resolution}px is incomplete (missing {missing}) "
                f"in {cache_dir}; available image arrays: {available}")

        images = np.load(paths["images"], mmap_mode="r")
        labels = np.load(paths["labels"])
        meta = CacheMetadata.from_dict(json.loads(paths["meta"].read_text()))

        # Validation here rather than at use: a mismatch discovered mid-benchmark produces a
        # plausible but wrong accuracy, whereas failing at open is unambiguous.
        if images.ndim != 4 or images.shape[3] != 3 or images.dtype != np.uint8:
            raise CacheError(f"expected uint8 NHWC, found {images.dtype} {images.shape}")
        if images.shape[1] != resolution or images.shape[2] != resolution:
            raise CacheError(
                f"cache holds {images.shape[1]}x{images.shape[2]}px images, "
                f"requested {resolution}px")
        if len(images) != len(labels):
            raise CacheError(f"{len(images)} images but {len(labels)} labels")
        if meta.n_samples != len(labels):
            raise CacheError(f"metadata claims {meta.n_samples} samples, found {len(labels)}")
        if meta.label_digest and meta.label_digest != _digest(labels):
            raise CacheError(
                "label digest mismatch: the label file has changed since the cache was built")
        # [AUDIT C5] Label dtype and range, and the metadata's own resolution and class count,
        # were never checked.
        # Before: (validation ended at the digest check)
        if labels.dtype != LABEL_DTYPE:
            raise CacheError(f"labels are {labels.dtype}, expected {np.dtype(LABEL_DTYPE)}")
        if meta.resolution != resolution:
            raise CacheError(f"metadata records {meta.resolution}px, requested {resolution}px")
        if len(meta.classes) != meta.n_classes:
            raise CacheError(f"metadata lists {len(meta.classes)} class names but "
                             f"n_classes={meta.n_classes}")
        if len(labels) and (labels.min() < 0 or labels.max() >= meta.n_classes):
            raise CacheError(f"labels span [{labels.min()}, {labels.max()}], outside "
                             f"[0, {meta.n_classes})")
        return cls(images, labels, meta)

    # ---- properties ---------------------------------------------------------------------

    @property
    def images(self) -> np.ndarray:
        """Memory-mapped image array.

        Returns:
            Read-only uint8 NHWC array. Slices must be copied before being handed to torch,
            which warns on read-only buffers.
        """
        return self._images

    @property
    def labels(self) -> np.ndarray:
        """Label array.

        Returns:
            int64 array of length n_samples.
        """
        return self._labels

    @property
    def classes(self) -> list[str]:
        """Class names indexed by label value.

        Returns:
            The class list from metadata.
        """
        return self.meta.classes

    @property
    def resolution(self) -> int:
        """Square side length of the cached images.

        Returns:
            The resolution in pixels.
        """
        return int(self._images.shape[1])

    def __len__(self) -> int:
        """Number of cached samples.

        Returns:
            The sample count.
        """
        return len(self._labels)

    # ---- access -------------------------------------------------------------------------

    def take(self, indices: np.ndarray,
             out: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Read a set of samples into memory.

        Args:
            indices: Indices to read. Sorting them first gives sequential access to the memory
                map, which is markedly faster than random access.
            out: Optional preallocated uint8 array of shape (len(indices), H, W, 3) to read
                into, for example a pinned host buffer, avoiding any intermediate copy.

        Returns:
            A writable uint8 NHWC image array (out, when given) and its int64 labels.

        Raises:
            CacheError: If any index is out of range.
        """
        idx = np.asarray(indices)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self._labels)):
            raise CacheError(
                f"indices span [{idx.min()}, {idx.max()}], cache holds {len(self._labels)}")
        # [AUDIT C4] Indexing with an array already returns a new, writable array; the extra
        # np.array copied every batch a second time. The read-only concern applies only to
        # basic slicing, handled in slice() below.
        # Before:
        #   # np.array forces a writable copy: a memory-map slice is read-only, and
        #   # torch.from_numpy on a read-only buffer warns and yields an unsafe tensor.
        #   return np.array(self._images[idx]), self._labels[idx]
        if out is not None:
            # mode="clip" skips take's internal buffering; bounds were checked above.
            np.take(self._images, idx, axis=0, out=out, mode="clip")
            return out, self._labels[idx]
        return np.asarray(self._images[idx]), self._labels[idx]

    def slice(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a contiguous range of samples.

        Args:
            start: First index, inclusive.
            stop: Last index, exclusive.

        Returns:
            A writable uint8 NHWC image array and its int64 labels.

        Raises:
            CacheError: If the range falls outside the cache.
        """
        # [AUDIT C8] numpy clips out-of-range slices silently, returning fewer items than asked.
        # Before: (no check)
        if not 0 <= start <= stop <= len(self._labels):
            raise CacheError(
                f"slice [{start}, {stop}) is outside the cache of {len(self._labels)}")
        # np.array forces a writable copy: a memory-map slice is read-only, and
        # torch.from_numpy on a read-only buffer warns and yields an unsafe tensor.
        return np.array(self._images[start:stop]), self._labels[start:stop]

    def describe(self) -> str:
        """Render a one-line summary for run logs.

        Returns:
            A compact description of the cache contents.
        """
        gib = self._images.nbytes / 2 ** 30
        return (f"{self.meta.dataset}/{self.meta.split} "
                f"{len(self)} images @ {self.resolution}px, "
                f"{len(self.classes)} classes, {gib:.2f} GiB")