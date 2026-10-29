import torch
from edgexbench.core.registry import get_registry
from edgexbench.models.external import register_all

register_all()
reg = get_registry()
print(reg.list_models("shufflenetv1*", source="external"))

r = reg.resolve("shufflenetv1_1_0x_g3", num_classes=100)
print(f"{sum(p.numel() for p in r.module.parameters())/1e6:.2f} M params")
r.module.eval()
print(r.module(torch.zeros(2, 3, 160, 160)).shape)   # must be (2, 100)  <- the AvgPool test
print(r.module(torch.zeros(2, 3, 224, 224)).shape)