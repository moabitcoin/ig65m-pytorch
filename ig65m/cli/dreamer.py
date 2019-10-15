import sys

import torch
import torch.nn as nn

from torchvision.transforms import Compose

import numpy as np
from PIL import Image
from tqdm import tqdm

from einops import rearrange
from einops.layers.torch import Rearrange

from ig65m.models import r2plus1d_34_32_ig65m
from ig65m.datasets import VideoDataset
from ig65m.transforms import ToTensor, Resize, Normalize, Denormalize


class VideoModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = r2plus1d_34_32_ig65m(num_classes=359, pretrained=True, progress=True)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        #x = self.model.layer3(x)
        #x = self.model.layer4(x)

        return x


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        loss = 0.

        loss += (inputs[:, :, :-1, :, :] - inputs[:, :, 1:, :, :]).abs().sum() / inputs.numel()
        loss += (inputs[:, :, :, :-1, :] - inputs[:, :, :, 1:, :]).abs().sum() / inputs.numel()
        loss += (inputs[:, :, :, :, :-1] - inputs[:, :, :, :, 1:]).abs().sum() / inputs.numel()

        return loss


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    model = VideoModel()
    model = model.to(device)
    model = nn.DataParallel(model)
    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    mean, std = [0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize(128),
        Normalize(mean=mean, std=std),
    ])

    variation = TotalVariationLoss()

    denormalize = Denormalize(mean=mean, std=std)

    dataset = VideoDataset(args.video, clip=32, transform=transform)

    video = next(iter(dataset))
    #video = torch.rand(3, 32, 128, 128)
    assert video.size()[0:2] == (3, 32)
    video = rearrange(video, "c t h w -> () c t h w")
    video = video.data.cpu().numpy()

    # leaf node with grads and on device
    video = torch.tensor(video, requires_grad=True, device=device)

    progress = tqdm(range(args.num_epochs))

    for epoch in progress:
        model.zero_grad()

        acts = model(video)

        loss = acts.norm() + (-1) * 100 * variation(video)
        loss.backward()

        avg = video.grad.data.abs().mean()
        lr = args.lr / avg

        video.data += lr * video.grad.data

        for i in range(video.size(1)):
            cmin = -mean[i] / std[i]
            cmax = (1 - mean[i]) / std[i]
            video.data[0, i].clamp_(cmin, cmax)

        video.grad.data.zero_()

        progress.set_postfix({"loss": loss.item(), "avg": avg.item(), "lr": lr.item()})

    video = rearrange(video, "() c t h w -> c t h w")
    video = denormalize(video)
    video = rearrange(video, "c t h w -> t h w c")
    video.clamp_(0, 1)
    video = video.data.cpu().numpy()

    assert video.shape[0] == 32
    assert video.shape[3] == 3

    assert video.dtype == np.float32
    assert (video >= 0).all()
    assert (video <= 1).all()

    video = (video * 255).astype(np.uint8)

    images = [Image.fromarray(v, mode="RGB") for v in video]

    images[0].save(args.dream, format="GIF", append_images=images[1:],
                   save_all=True, duration=(1000 / 30), loop=0)

    print("💤 Done", file=sys.stderr)


