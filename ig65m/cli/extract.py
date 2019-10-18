import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

import numpy as np
from einops.layers.torch import Rearrange, Reduce
from tqdm import tqdm

from ig65m.models import r2plus1d_34_32_ig65m
from ig65m.datasets import VideoDataset
from ig65m.transforms import ToTensor, Resize, Normalize


class VideoModel(nn.Module):
    def __init__(self, pool_spatial="mean", pool_temporal="mean"):
        super().__init__()

        self.model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)

        self.pool_spatial = Reduce("n c t h w -> n c t", reduction=pool_spatial)
        self.pool_temporal = Reduce("n c t -> n c", reduction=pool_temporal)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.pool_spatial(x)
        x = self.pool_temporal(x)

        return x


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    model = VideoModel(pool_spatial=args.pool_spatial,
                       pool_temporal=args.pool_temporal)

    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    model = model.to(device)
    model = nn.DataParallel(model)

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize(args.frame_size),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    ])

    # dataset = WebcamDataset(clip=32, transform=transform)

    dataset = VideoDataset(args.video, clip=32, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    features = []

    with torch.no_grad():
        for inputs in tqdm(loader, total=len(dataset) // args.batch_size):
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs = outputs.data.cpu().numpy()

            for output in outputs:
                features.append(output)

    np.save(args.features, np.array(features), allow_pickle=False)
    print("🍪 Done", file=sys.stderr)
