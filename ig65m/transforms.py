import torch
import torch.nn as nn

import numpy as np


class ToTensor:
    def __call__(self, x):
        return torch.from_numpy(np.array(x)).float() / 255.


class Resize:
    def __init__(self, size, mode="bilinear"):
        self.size = size
        self.mode = mode

    def __call__(self, video):
        size = self.size
        scale = None

        if isinstance(size, int):
            scale = float(size) / min(video.shape[-2:])
            size = None

        return nn.functional.interpolate(video, size=size, scale_factor=scale,
                                         mode=self.mode, align_corners=False)


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, video):
        size = self.size

        if isinstance(size, int):
            size = size, size

        th, tw = size
        h, w = video.shape[-2:]

        i = int(round((h - th) / 2.))
        j = int(round((w - tw) / 2.))

        return video[..., i:(i + th), j:(j + tw)]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video - mean) / std


class Denormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, video):
        shape = (-1,) + (1,) * (video.dim() - 1)

        mean = torch.as_tensor(self.mean, device=video.device).reshape(shape)
        std = torch.as_tensor(self.std, device=video.device).reshape(shape)

        return (video * std) + mean
