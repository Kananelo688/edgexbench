# ShuffleNetV1
Repo:    https://github.com/megvii-model/ShuffleNet-Series
Commit:  aa91feb71b01f28d0b8da3533d20a3edb11b1810
Path:    ShuffleNetV1/
Copied:  2026-10-29
Files:   network.py, blocks.py
License: MIT
Paper:   Zhang et al., CVPR 2018, arXiv:1707.01083

## Modifications
- `from blocks import ...` -> `from .blocks import ...`
- `nn.AvgPool2d(7)` -> `nn.AdaptiveAvgPool2d(1)`: upstream hard-codes a 224px input,
  which fails at the 160px training resolution. Identical behaviour at 224.
- commented `print('model size is ', model_size)`
