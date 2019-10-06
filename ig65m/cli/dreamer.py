import sys

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from ig65m.models import r2plus1d_34_32_ig65m
from ig65m.transforms import Denormalize


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    dream = torch.rand(1, 3, 32, 112, 112, requires_grad=True, device=device)

    criterion = ElectricSheepLoss(device)
    regularize = TotalVariationLoss()

    optimizer = torch.optim.Adam([dream], lr=args.lr)

    mean, std = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]
    denormalize = Denormalize(mean=mean, std=std)

    progress = tqdm(range(args.num_epochs))

    for epoch in progress:
        video = dream.clamp(-1, 1)

        optimizer.zero_grad()

        act = criterion(video)
        reg = regularize(video) * args.gamma

        loss = act + reg

        loss.backward()
        optimizer.step()

        progress.set_postfix({"loss": loss.item(), "act": act.item(), "reg": reg.item()})

    dream = dream.clamp(-1, 1)
    dream = rearrange(dream, "() c t h w -> c t h w")
    dream = denormalize(dream)
    dream = rearrange(dream, "c t h w -> t h w c")
    dream = dream.data.cpu().numpy()

    assert dream.shape == (32, 112, 112, 3)
    assert dream.dtype == np.float32
    assert (dream >= 0).all()
    assert (dream <= 1).all()

    dream = (dream * 255).astype(np.uint8)

    images = [Image.fromarray(v, mode="RGB") for v in dream]

    images[0].save(args.image, format="GIF", append_images=images[1:], save_all=True, duration=(1000 / 30), loop=0)

    print("💤 Done", file=sys.stderr)


class ElectricSheepLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)

        for params in model.parameters():
            params.requires_grad = False

        model = model.to(device)
        model = nn.DataParallel(model)
        model.eval()

        self.model = model

    def forward(self, inputs):
        x = self.model.module.stem(inputs)
        #x = self.model.module.layer1(x)
        #x = self.model.module.layer2(x)
        #x = self.model.module.layer3(x)
        #x = self.model.module.layer4(x)

        # TODO: pick i to maximize x[:, i, :, :, :]

        loss = (-1 * x[:, :, :, :, :]).mean()

        return loss


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        loss = 0.

        loss += (inputs[:, :, :-1, :, :] - inputs[:, :, 1:, :, :]).abs().sum() / inputs.numel()
        loss += (inputs[:, :, :, :-1, :] - inputs[:, :, :, 1:, :]).abs().sum() / inputs.numel()
        loss += (inputs[:, :, :, :, :-1] - inputs[:, :, :, :, 1:]).abs().sum() / inputs.numel()

        return loss
