"""Trainer behaviour: accumulation step counts, resume, and input validation (audit T1-T11)."""

from __future__ import annotations

import pytest
import torch

from edgebench.core.artifacts import load_checkpoint
from edgebench.core.config import RecipeConfig, RunConfig
from edgebench.core.registry import ModelSpec
from edgebench.train.trainer import Trainer, TrainingError

from helpers import FakeCache, tiny_registry

RES = 16


def recipe(**changes) -> RecipeConfig:
    base = dict(num_classes=4, train_res=RES, train_cache_res=RES, eval_res=RES, epochs=2,
                warmup_epochs=0, batch_size=8, mixup_alpha=0.0, cutmix_alpha=0.0,
                amp_dtype="float32", seed=0)
    base.update(changes)
    return RecipeConfig(**base)


def run(**changes) -> RunConfig:
    base = dict(device="cpu", micro_batch_size=2, channels_last=False)
    base.update(changes)
    return RunConfig(**base)


def make(rec, rn, tmp_path=None, train=None, val=None, spec="tiny", **kw) -> Trainer:
    return Trainer(spec, rec, rn, train or FakeCache(64, RES), val or FakeCache(16, RES),
                   registry=tiny_registry(),
                   resume_path=(tmp_path / "tiny.resume.pt") if tmp_path else None, **kw)


@pytest.mark.parametrize("limit, steps_per_epoch", [(3, 1), (5, 2), (8, 2)])
def test_accumulation_steps_every_group_including_partial(tmp_path, limit, steps_per_epoch):
    # accum = 8 / 2 = 4 micro-batches per optimiser step.
    make(recipe(), run(limit_batches=limit), tmp_path).fit()
    ts = load_checkpoint(tmp_path / "tiny.resume.pt")["train_state"]
    assert ts["step"] == 2 * steps_per_epoch


def test_resume_at_final_epoch_returns_result(tmp_path):
    first = make(recipe(), run(limit_batches=4), tmp_path).fit()
    again = make(recipe(), run(limit_batches=4), tmp_path).fit()
    assert again.epochs_completed == 2
    assert again.final.top1 == pytest.approx(first.final.top1)


def test_resume_reproduces_uninterrupted_run(tmp_path):
    full = make(recipe(), run(limit_batches=4)).fit()

    def crash_at_epoch_2(rec):
        if rec.epoch == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        make(recipe(), run(limit_batches=4), tmp_path, on_epoch=crash_at_epoch_2).fit()
    resumed = make(recipe(), run(limit_batches=4), tmp_path).fit()
    assert resumed.final.train_loss == pytest.approx(full.final.train_loss, rel=1e-6)


def test_class_list_mismatch_rejected():
    val = FakeCache(16, RES)
    val.classes = list(reversed(val.classes))
    with pytest.raises(TrainingError, match="class"):
        make(recipe(), run(), val=val)


def test_odd_micro_batch_with_mixing_rejected():
    with pytest.raises(TrainingError, match="even"):
        make(recipe(batch_size=9, mixup_alpha=0.2), run(micro_batch_size=3))


def test_string_spec_accepted():
    result = make(recipe(epochs=1), run(limit_batches=4), spec="tiny").fit()
    assert result.model == "tiny"


def test_sgd_without_momentum_builds():
    make(recipe(optimizer="sgd", momentum=0.0, epochs=1), run(limit_batches=4)).fit()


def test_tuple_output_fails_loudly():
    class TwoHeads(torch.nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.fc = torch.nn.Linear(3, num_classes)

        def forward(self, x):
            y = self.fc(x.mean((2, 3)))
            return y, y

    reg = tiny_registry()
    reg.register_external("twoheads", TwoHeads, repo="tests", commit="none")
    t = Trainer(ModelSpec("twoheads"), recipe(epochs=1), run(limit_batches=4),
                FakeCache(64, RES), FakeCache(16, RES), registry=reg)
    with pytest.raises(TrainingError, match="tuple"):
        t.fit()
