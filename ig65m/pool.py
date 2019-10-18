import torch.nn as nn

from einops import reduce


def adaptive_sum_pool3d(x, output_size):
    if isinstance(output_size, int):
        output_size = (output_size,) * 3

    t, h, w = output_size

    return reduce(x, "n c (t t2) (h h2) (w w2) -> n c t h w", "sum", t=t, h=h, w=w)


class AdaptiveSumPool3d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_sum_pool3d(x, self.output_size)
