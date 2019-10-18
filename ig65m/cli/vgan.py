import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms import Compose

from einops import rearrange
from einops.layers.torch import Rearrange

from ig65m.datasets import VideoDirectoryDataset
from ig65m.transforms import ToTensor, Resize, CenterCrop, Normalize, Denormalize
from ig65m.losses import GeneratorHingeLoss, DiscriminatorHingeLoss
from ig65m.gan import Generator, Discriminator


def main(args):
    if torch.cuda.is_available():
        print("🐎 Running on GPU(s)", file=sys.stderr)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        print("🐌 Running on CPU(s)", file=sys.stderr)
        device = torch.device("cpu")

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    transform = Compose([
        ToTensor(),
        Rearrange("t h w c -> c t h w"),
        #Resize(48),
        CenterCrop(32),
        Normalize(mean=mean, std=std),
    ])

    denormalize = Denormalize(mean=mean, std=std)

    dataset = VideoDirectoryDataset(args.videos, clip_length=args.clip_length, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    args.checkpoints.mkdir(exist_ok=True)

    g = Generator(args.z_dimension)
    g = g.to(device)
    g = nn.DataParallel(g)

    d = Discriminator()
    d = d.to(device)
    d = nn.DataParallel(d)

    lr_g = 1e-4 * 1
    lr_d = 1e-4 * 4

    opt_g = torch.optim.Adam([p for p in g.parameters() if p.requires_grad],
                             lr=lr_g, betas=(0, 0.9))

    opt_d = torch.optim.Adam([p for p in d.parameters() if p.requires_grad],
                             lr=lr_d, betas=(0, 0.9))

    crit_g = GeneratorHingeLoss()
    crit_d = DiscriminatorHingeLoss()

    zfix = torch.randn(1, args.z_dimension, device=device).clamp_(0)

    step = 0

    with SummaryWriter(str(args.logs)) as summary:
        for _ in range(args.num_epochs):
            for inputs in loader:
                adjust_learning_rate(opt_g, step, lr_g)
                adjust_learning_rate(opt_d, step, lr_d)

                # Step D

                g.zero_grad()
                d.zero_grad()

                z = torch.randn(inputs.size(0), args.z_dimension, device=device).clamp_(0)

                real_data = inputs.to(device)
                fake_data = g(z)

                real_out = d(real_data)
                fake_out = d(fake_data)

                loss_d_real, loss_d_fake = crit_d(real_out, fake_out)
                loss_d = loss_d_real.mean() + loss_d_fake.mean()
                loss_d.backward()

                opt_d.step()

                # Step G

                g.zero_grad()
                d.zero_grad()

                z = torch.randn(inputs.size(0), args.z_dimension, device=device).clamp_(0)

                fake_data = g(z)
                fake_out = d(fake_data)

                loss_g = crit_g(fake_out)
                loss_g.backward()

                opt_g.step()

                # Done

                summary.add_scalar("Loss/Discriminator/Real", loss_d_real.item(), step)
                summary.add_scalar("Loss/Discriminator/Fake", loss_d_fake.item(), step)
                summary.add_scalar("Loss/Generator", loss_g.item(), step)

                if step % args.save_frequency == 0:
                    real_data = inputs
                    real_clip = denormalize(real_data[0])
                    real_images = rearrange(real_clip, "c t h w -> t c h w")

                    summary.add_images("Images/Real", real_images, step)

                    with torch.no_grad():
                        for m in g.modules():
                            if isinstance(m, nn.BatchNorm3d):
                                m.eval()

                        fake_data = g(zfix)

                        for m in g.modules():
                            if isinstance(m, nn.BatchNorm3d):
                                m.train()

                    fake_clip = denormalize(fake_data[0])
                    fake_images = rearrange(fake_clip, "c t h w -> t c h w")

                    summary.add_images("Images/Fake", fake_images, step)

                    state = {"step": step,
                             "g_state_dict": g.state_dict(), "d_state_dict": d.state_dict(),
                             "g_opt": opt_g.state_dict(), "d_opt": opt_d.state_dict()}

                    torch.save(state, args.checkpoints / "state-{:010d}.pth".format(step))

                step += 1

    print("🥑 Done", file=sys.stderr)


# https://arxiv.org/abs/1706.02677
def adjust_learning_rate(optimizer, step, lr):
    warmup = 1000
    base = 0.01 * lr

    def lerp(c, first, last):
        return first + c * (last - first)

    if step <= warmup:
        lr = lerp(step / warmup, base, lr)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
