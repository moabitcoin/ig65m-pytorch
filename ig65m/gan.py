import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from ig65m.upsample import Upsample3d
from ig65m.attention import GlobalContext3d
from ig65m.pool import AdaptiveSumPool3d


# BigGan building blocks; see paper
# https://arxiv.org/abs/1809.11096


class Generator(nn.Module):
    def __init__(self, z):
        super().__init__()

        ch = 32
        T, H, W = 2, 2, 2

        self.project = nn.Sequential(
            nn.Linear(z, 16 * ch * T * H * W),
            Rearrange("n (c t h w) -> n c t h w", t=T, h=H, w=W))

        self.layer1 = ResBlockUp(16 * ch, 8 * ch)
        self.layer2 = ResBlockUp(8 * ch, 4 * ch)
        self.layer3 = ResBlockUp(4 * ch, 2 * ch)
        self.layer4 = ResBlockUp(2 * ch, ch)

        self.ctx1 = GlobalContext3d(4 * ch)
        self.ctx2 = GlobalContext3d(2 * ch)

        self.final = nn.Sequential(
            nn.BatchNorm3d(ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch, 3, kernel_size=3, padding=1, bias=True))

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.utils.spectral_norm(m)

    def forward(self, x):
        x = self.project(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.ctx1(x)
        x = self.layer3(x)
        x = self.ctx2(x)
        x = self.layer4(x)

        x = self.final(x)
        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        ch = 32

        self.layer1 = ResBlockDown(3, 2 * ch)
        self.layer2 = ResBlockDown(2 * ch, 4 * ch)
        self.layer3 = ResBlockDown(4 * ch, 8 * ch)
        self.layer4 = ResBlockDown(8 * ch, 8 * ch)

        self.layer5 = ResBlockDown(8 * ch, 8 * ch, down=None)

        self.ctx1 = GlobalContext3d(2 * ch)
        self.ctx2 = GlobalContext3d(4 * ch)

        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            AdaptiveSumPool3d(1),
            Rearrange("n c () () () -> n c"),
            nn.Linear(8 * ch, 1))

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.utils.spectral_norm(m)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ctx1(x)
        x = self.layer2(x)
        x = self.ctx2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final(x)

        return x


class ResBlockUp(nn.Module):
    def __init__(self, inplanes, outplanes, up=(2, 2, 2)):
        super().__init__()

        self.up = up

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.bn2 = nn.BatchNorm3d(outplanes)

        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False)

    def forward(self, x):
        xx = self.bn1(x)
        xx = nn.functional.relu(xx, inplace=True)
        xx = nn.functional.interpolate(xx, scale_factor=self.up, mode="nearest")
        xx = self.conv1(xx)
        xx = self.bn2(xx)
        xx = nn.functional.relu(xx, inplace=True)
        xx = self.conv2(xx)

        x = nn.functional.interpolate(x, scale_factor=self.up, mode="nearest")
        x = self.conv3(x)

        return x + xx


class ResBlockDown(nn.Module):
    def __init__(self, inplanes, outplanes, down=(2, 2, 2)):
        super().__init__()

        self.down = down

        self.conv1 = nn.Conv3d(inplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv3d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=False)

    def forward(self, x):
        xx = nn.functional.relu(x, inplace=False)
        xx = self.conv1(xx)
        xx = nn.functional.relu(xx, inplace=True)
        xx = self.conv2(xx)

        if self.down is not None:
            xx = nn.functional.avg_pool3d(xx, self.down)

        x = self.conv3(x)

        if self.down is not None:
            x = nn.functional.avg_pool3d(x, self.down)

        return x + xx
