"""Config typing, results round-trips, registry and artifact helpers (audit F1-F2, T3, R1, R3,
G1, T9, A1)."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from edgexbench.core.artifacts import strip_wrapper_prefixes
from edgexbench.core.config import ConfigError, RecipeConfig, RunConfig
from edgexbench.core.registry import ModelRegistry, TimmProvider, get_registry
from edgexbench.core.results import STATUS_COMPLETED, ResultsLayout, ResultsStore


def test_yaml_scientific_notation_is_a_float():
    data = yaml.safe_load("lr: 2e-3")
    assert RecipeConfig.from_dict(data).lr == pytest.approx(0.002)


def test_int_and_float_spellings_hash_identically():
    assert RecipeConfig(lr=1, min_lr=0).digest() == RecipeConfig(lr=1.0, min_lr=0.0).digest()


@pytest.mark.parametrize("field, value", [("lr", "fast"), ("epochs", 100.5), ("epochs", True)])
def test_bad_types_raise_config_error(field, value):
    with pytest.raises(ConfigError):
        RecipeConfig(**{field: value})


def test_float16_rejected():
    with pytest.raises(ConfigError):
        RecipeConfig(amp_dtype="float16")


def test_run_config_overrides_are_validated():
    with pytest.raises(ConfigError):
        dataclasses.replace(RunConfig(), limit_batches=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        RunConfig().limit_batches = 0


def store(tmp_path) -> ResultsStore:
    return ResultsStore(ResultsLayout(results_dir=tmp_path / "r", artifact_dir=tmp_path / "a"))


@pytest.mark.parametrize("name, digest", [("m", "012345678901"), ("100", "abcdefabcdef")])
def test_is_done_survives_csv_type_inference(tmp_path, name, digest):
    s = store(tmp_path)
    s.artifact_path(name).parent.mkdir(parents=True)
    s.artifact_path(name).write_bytes(b"x")
    s.record({"model": name, "status": STATUS_COMPLETED, "recipe_digest": digest})
    assert s.is_done(name, digest)


def test_truncate_epochs_only_touches_the_given_digest(tmp_path):
    s = store(tmp_path)
    for digest in ("aaa", "bbb"):
        for epoch in (1, 2, 3):
            s.log_epoch({"model": "m", "epoch": epoch, "recipe_digest": digest})
    s.truncate_epochs("m", after_epoch=1, digest="bbb")
    df = s.epochs()
    assert len(df[df.recipe_digest == "aaa"]) == 3
    assert len(df[df.recipe_digest == "bbb"]) == 1


def test_external_registration_visible_with_custom_providers():
    import torch
    reg = ModelRegistry(providers=[TimmProvider()])
    reg.register_external("mynet", lambda num_classes: torch.nn.Linear(1, num_classes),
                          repo="x", commit="y")
    assert reg.is_available("mynet")


def test_torchvision_inception_forced_single_head_and_recorded():
    resolved = get_registry().resolve({"name": "googlenet", "source": "torchvision",
                                       "kwargs": {"init_weights": False}}, 10)
    assert resolved.kwargs["aux_logits"] is False


def test_nested_wrapper_prefixes_stripped():
    assert list(strip_wrapper_prefixes({"module._orig_mod.fc.weight": 1})) == ["fc.weight"]
