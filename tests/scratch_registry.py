"""Scratch: poke at the registry interactively. Not a test; prints, does not assert."""
from edgexbench.core.registry import ModelRegistry

reg = ModelRegistry()

print(reg.describe("mobilenetv2_100"))
print(reg.describe("inception_v3"))        # two candidates — see the shadowing
print(reg.list_models("*ghostnet*"))

r = reg.resolve("mobilenetv2_100", num_classes=100)
print(r.provenance())
print(sum(p.numel() for p in r.module.parameters()) / 1e6, "M params")