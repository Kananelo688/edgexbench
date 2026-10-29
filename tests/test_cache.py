"""Cache discovery and build safety (audit C1-C7)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from edgexbench.core.data.cache import (CacheError, ImageCache, build_cache, cache_exists,
                                       cache_paths, discover_class_folders)

RES = 8


def write_png(path, value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((12, 12, 3), value, np.uint8)).save(path)


def square(img):
    return np.asarray(img.resize((RES, RES)), dtype=np.uint8)


def test_kaggle_layout_from_dataset_root_keeps_classes_and_excludes_val(tmp_path):
    for part, cls in [("train.X1", "n01"), ("train.X1", "n02"), ("train.X2", "n03")]:
        write_png(tmp_path / part / cls / "a.JPEG")
    write_png(tmp_path / "val.X" / "n01" / "v.JPEG")
    found = discover_class_folders([tmp_path])
    assert found.classes == ["n01", "n02", "n03"]
    assert len(found) == 3


def test_kaggle_layout_from_part_roots_merges_classes(tmp_path):
    write_png(tmp_path / "train.X1" / "n01" / "a.png")
    write_png(tmp_path / "train.X2" / "n01" / "b.png")
    found = discover_class_folders([tmp_path / "train.X1", tmp_path / "train.X2"])
    assert found.classes == ["n01"] and len(found) == 2


def test_nested_layout_with_label_from_top(tmp_path):
    write_png(tmp_path / "n01" / "images" / "a.png")
    write_png(tmp_path / "n02" / "images" / "b.png")
    assert discover_class_folders([tmp_path], label_from="top").classes == ["n01", "n02"]


def test_files_directly_in_root_and_hidden_files_skipped(tmp_path):
    write_png(tmp_path / "n01" / "a.png")
    write_png(tmp_path / "stray.png")
    (tmp_path / "n01" / "._a.png").write_bytes(b"not an image")
    found = discover_class_folders([tmp_path])
    assert found.classes == ["n01"] and len(found) == 1


def test_duplicate_canonical_classes_rejected(tmp_path):
    write_png(tmp_path / "n01" / "a.png")
    with pytest.raises(CacheError, match="duplicate"):
        discover_class_folders([tmp_path], classes=["n01", "n01"])


def build(tmp_path, samples, transform=square, **kw):
    return build_cache(samples, transform, tmp_path / "cache", "train", RES, dataset="t",
                       classes=["n01"], progress=False, workers=2, **kw)


def test_interrupted_rebuild_leaves_no_complete_cache(tmp_path):
    files = [tmp_path / f"n01/{i}.png" for i in range(4)]
    for f in files:
        write_png(f)
    samples = [(f, 0) for f in files]
    build(tmp_path, samples)

    def explode(img):
        raise RuntimeError("boom")
    with pytest.raises(CacheError):
        build(tmp_path, samples, transform=explode, overwrite=True)
    assert not cache_exists(tmp_path / "cache", "train", RES)
    assert not list((tmp_path / "cache").glob("*.tmp*"))


def test_decode_error_names_the_file(tmp_path):
    bad = tmp_path / "n01" / "bad.png"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"corrupt")
    with pytest.raises(CacheError, match="bad.png"):
        build(tmp_path, [(bad, 0)])


def test_take_returns_writable_correct_rows(tmp_path):
    files = [tmp_path / f"n01/{i}.png" for i in range(4)]
    for i, f in enumerate(files):
        write_png(f, value=10 * i)
    build(tmp_path, [(f, 0) for f in files])
    cache = ImageCache.open(tmp_path / "cache", "train", RES)
    images, labels = cache.take(np.array([1, 3]))
    assert images.flags.writeable
    assert images[0, 0, 0, 0] == 10 and images[1, 0, 0, 0] == 30
    assert cache_paths(tmp_path / "cache", "train", RES)["images"].exists()
