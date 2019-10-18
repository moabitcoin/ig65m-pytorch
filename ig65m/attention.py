import torch
import torch.nn as nn

from einops import rearrange


# Attention: start with this paper
# https://arxiv.org/abs/1904.11492


class SelfAttention3d(nn.Module):
    def __init__(self, planes):
        super().__init__()

        # Note: ratios below should be made configurable

        self.q = nn.Conv3d(planes, planes // 8, kernel_size=1, bias=False)
        self.k = nn.Conv3d(planes, planes // 8, kernel_size=1, bias=False)
        self.v = nn.Conv3d(planes, planes // 2, kernel_size=1, bias=False)
        self.z = nn.Conv3d(planes // 2, planes, kernel_size=1, bias=False)

        self.y = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # Note: pooling below should be made configurable

        k = nn.functional.max_pool3d(k, (2, 2, 2))
        v = nn.functional.max_pool3d(v, (2, 2, 2))

        q = rearrange(q, "n c t h w -> n (t h w) c")
        k = rearrange(k, "n c t h w -> n c (t h w)")
        v = rearrange(v, "n c t h w -> n c (t h w)")

        beta = torch.bmm(q, k)
        beta = torch.softmax(beta, dim=-1)
        beta = rearrange(beta, "n thw c -> n c thw")

        att = torch.bmm(v, beta)
        att = rearrange(att, "n c (t h w) -> n c t h w",
                        t=x.size(2), h=x.size(3), w=x.size(4))

        return self.y * self.z(att) + x


class SimpleSelfAttention3d(nn.Module):
    def __init__(self, planes):
        super().__init__()

        self.k = nn.Conv3d(planes, 1, kernel_size=1, bias=False)
        self.v = nn.Conv3d(planes, planes, kernel_size=1, bias=False)

        self.y = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        k = self.k(x)
        k = rearrange(k, "n c t h w -> n (t h w) c")
        k = torch.softmax(k, dim=-1)

        xx = rearrange(x, "n c t h w -> n c (t h w)")

        ctx = torch.bmm(xx, k)
        ctx = rearrange(ctx, "n c () -> n c () () ()")

        att = self.v(ctx)

        return self.y * att + x


class GlobalContext3d(nn.Module):
    def __init__(self, planes):
        super().__init__()

        self.k = nn.Conv3d(planes, 1, kernel_size=1, bias=False)

        # Note: ratios below should be made configurable

        self.v = nn.Sequential(
            nn.Conv3d(planes, planes // 8, kernel_size=1, bias=False),
            nn.LayerNorm((planes // 8, 1, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(planes // 8, planes, kernel_size=1, bias=False))

        self.y = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        k = self.k(x)
        k = rearrange(k, "n c t h w -> n (t h w) c")
        k = torch.softmax(k, dim=-1)

        xx = rearrange(x, "n c t h w -> n c (t h w)")

        ctx = torch.bmm(xx, k)
        ctx = rearrange(ctx, "n c () -> n c () () ()")

        att = self.v(ctx)

        return self.y * att + x
