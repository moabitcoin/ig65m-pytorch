import sys

import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from einops import rearrange

from ig65m.models import r2plus1d_34_32_ig65m


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


    for epoch in tqdm(range(args.num_epochs)):
        optimizer.zero_grad()

        loss = criterion(dream)

        print("epoch: {} loss: {}".format(epoch, loss.item()))

        loss.backward()
        optimizer.step()

    # dream = rearrange(dream, "() c t h w -> t h w c")
    # for image in dream:
    #     image = Image.fromarray(image, mode="RGB")
    #
    # TODO: how to animate?
    #
    # image.save(args.image, optimize=True)

    print("💤 Done", file=sys.stderr)


class ElectricSheepLoss(nn.Module):
    def __init__(self, device):
        super().__init__()

        model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)

        model = model.to(device)
        model = nn.DataParallel(model)
        model.eval()

        self.model = model

    def forward(self, inputs):
        x = self.model.module.stem(inputs)

        return (-1. * x).mean()
