import sys

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

from ig65m.models import r2plus1d_34_32_ig65m
from ig65m.transforms import Normalize


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
    optimizer = torch.optim.Adam([dream], lr=1e-4)

    normalize = Normalize(mean=[0.43216, 0.394666, 0.37645],
                          std=[0.22803, 0.22145, 0.216989])

    progress = tqdm(range(args.num_epochs))

    for epoch in progress:
        optimizer.zero_grad()

        rgb = dream.clamp(min=0, max=1)
        rgb = normalize(rgb)

        loss = criterion(rgb)

        progress.set_postfix({"epoch": epoch, "loss": loss.item()})

        loss.backward()
        optimizer.step()


    dream = rearrange(dream, "() c t h w -> t h w c")
    dream = dream.clamp(min=0, max=1)
    dream = dream.data.cpu().numpy()

    assert (dream >= 0).all()
    assert (dream <= 1).all()

    dream = (dream * 255).astype(np.uint8)

    images = [Image.fromarray(v, mode="RGB") for v in dream]

    images[0].save(args.image, format="GIF", append_images=images[1:], save_all=True, duration=1000, loop=1)

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

        return (-1. * x).mean()
