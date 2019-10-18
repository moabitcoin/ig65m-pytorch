import torch
import torch.nn as nn

from einops import rearrange


# PixelShuffle and ICNR init paper
# https://arxiv.org/abs/1707.02937


class Upsample3d(nn.Module):
    def __init__(self, planes, upscale_factor):
        super().__init__()

        self.explode = nn.Conv3d(planes, planes * (upscale_factor ** 3),
                                 kernel_size=1, bias=False)

        self.shuffle = PixelShuffle3d(upscale_factor)

        icnr3d(self.explode.weight.data, upscale_factor)

    def forward(self, x):
        x = self.explode(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.shuffle(x)
        return x


def icnr3d(x, upscale_factor):
    n, c, t, h, w = x.size()
    g = int(n / (upscale_factor ** 3))

    z = torch.empty([g, c, t, h, w])
    z = nn.init.kaiming_normal_(z)
    z = rearrange(z, "g c t h w -> c g (t h w)", c=g, g=c)
    z = z.repeat(1, 1, upscale_factor ** 3)
    z = z.view([c, n, t, h, w])
    z = rearrange(z, "c n t h w -> n c t h w")

    x.copy_(z)


def pixel_shuffle3d(x, upscale_factor):
    return rearrange(x, "n (c t2 h2 w2) t h w -> n c (t t2) (h h2) (w w2)",
                     h2=upscale_factor, w2=upscale_factor, t2=upscale_factor,
                     c=x.size(1) // (upscale_factor ** 3))


class PixelShuffle3d(nn.Module):
    def __init__(self, upscale_factor):
        super().__init__()

        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle3d(x, self.upscale_factor)
