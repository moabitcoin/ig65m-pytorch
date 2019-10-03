import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.transforms import Compose

from ig65m.models import r2plus1d_34
from ig65m.datasets import VideoDataset, WebcamDataset
from ig65m.transforms import ToTensor, Resize, CenterCrop, Normalize

from einops.layers.torch import Rearrange


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

    # dataset = WebcamDataset(args.frames, transform=transform)

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
