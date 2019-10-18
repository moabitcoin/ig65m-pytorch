import torch.nn as nn


# Hinge losses for unconditioned
# GAN generator and discriminator
# https://arxiv.org/abs/1705.02894


class GeneratorHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake):
        return (-1.) * fake.mean()


class DiscriminatorHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real, fake):
        r = nn.functional.relu(1. - real).mean()
        f = nn.functional.relu(1. + fake).mean()
        return r, f
