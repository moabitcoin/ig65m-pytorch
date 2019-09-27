#!/usr/bin/env python3

import sys
import math
import json
import argparse
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from torchvision.transforms import Compose
from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D

from einops.layers.torch import Rearrange


def r2plus1d_34(num_classes):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    return model


class FrameRange:
    def __init__(self, video, first, last):
        assert first <= last

        for i in range(first):
            ret, _ = video.read()

            if not ret:
                raise RuntimeError("seeking to frame at index {} failed".format(i))

        self.video = video
        self.it = first
        self.last = last

    def __next__(self):
        if self.it >= self.last or not self.video.isOpened():
            raise StopIteration

        ok, frame = self.video.read()

        if not ok:
            raise RuntimeError("decoding frame at index {} failed".format(self.it))

        self.it += 1

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class BatchedRange:
    def __init__(self, rng, n):
        self.rng = rng
        self.n = n

    def __next__(self):
        ret = []

        for i in range(self.n):
            ret.append(next(self.rng))

        return ret


class TransformedRange:
    def __init__(self, rng, fn):
        self.rng = rng
        self.fn = fn

    def __next__(self):
        return self.fn(next(self.rng))


class VideoDataset(IterableDataset):
    def __init__(self, path, clip, transform=None):
        super().__init__()

        self.path = path
        self.clip = clip
        self.transform = transform

        video = cv2.VideoCapture(str(path))
        frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        self.first = 0
        self.last = frames

    def __iter__(self):
        info = get_worker_info()

        video = cv2.VideoCapture(str(self.path))

        if info is None:
            rng = FrameRange(video, self.first, self.last)
        else:
            per = int(math.ceil((self.last - self.first) / float(info.num_workers)))
            wid = info.id

            first = self.first + wid * per
            last = min(first + per, self.last)

            rng = FrameRange(video, first, last)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v

        return TransformedRange(BatchedRange(rng, self.clip), fn)


class WebcamDataset(IterableDataset):
    def __init__(self, clip, transform=None):
        super().__init__()

        self.clip = clip
        self.transform = transform
        self.video = cv2.VideoCapture(0)

    def __iter__(self):
        info = get_worker_info()

        if info is not None:
            raise RuntimeError("multiple workers not supported in WebcamDataset")

        # treat webcam as fixed frame range for now: 10 minutes
        rng = FrameRange(self.video, 0, 30 * 60 * 10)

        if self.transform is not None:
            fn = self.transform
        else:
            fn = lambda v: v

        return TransformedRange(BatchedRange(rng, self.clip), fn)


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video):
        return torch.nn.functional.interpolate(video, size=self.size,
            mode=self.mode, align_corners=False)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        h, w = video.shape[-2:]
        th, tw = self.size

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean).reshape(shape)
        std = torch.as_tensor(self.std).reshape(shape)

        return (video - mean) / std


def main(args):
    if args.labels:
        with args.labels.open() as f:
            labels = json.load(f)
    else:
        labels = list(range(args.classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = r2plus1d_34(num_classes=args.classes)
    model = model.to(device)

    weights = torch.load(args.model, map_location=device)
    model.load_state_dict(weights)

    model = nn.DataParallel(model)
    model.eval()

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        Resize((128, 171)),
        Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        CenterCrop((112, 112)),
    ])

    #dataset = WebcamDataset(args.frames, transform=transform)

    dataset = VideoDataset(args.video, args.frames, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for inputs in loader:
        # NxCxTxHxW
        assert inputs.size() == (args.batch_size, 3, args.frames, 112, 112)

        inputs = inputs.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, dim=1)
        preds = preds.data.cpu().numpy()

        scores = nn.functional.softmax(outputs, dim=1)
        scores = scores.data.cpu().numpy()

        for pred, score in zip(preds, scores):
            index = pred.item()
            label = labels[index]
            score = round(score.max().item(), 3)

            print("label='{}' score={}".format(label, score), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg("model", type=Path, help=".pth file to load model weights from")
    arg("video", type=Path, help="video file to run feature extraction on")
    arg("--frames", type=int, choices=(8, 32), required=True, help="clip frames for video model")
    arg("--classes", type=int, choices=(400, 487), required=True, help="classes in last layer")
    arg("--batch-size", type=int, default=1, help="number of sequences per batch for inference")
    arg("--num-workers", type=int, default=0, help="number of workers for data loading")
    arg("--labels", type=Path, help="JSON file with label map array")

    main(parser.parse_args())
