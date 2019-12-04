import torch.hub
import torch.nn as nn

from torchvision.models.video.resnet import VideoResNet, BasicBlock, R2Plus1dStem, Conv2Plus1D


model_urls = {
    "r2plus1d_34_8_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth",           # noqa: E501
    "r2plus1d_34_32_ig65m": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch-449a7af9.pth",         # noqa: E501
    "r2plus1d_34_8_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth",    # noqa: E501
    "r2plus1d_34_32_kinetics": "https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth",  # noqa: E501
}


def r2plus1d_34_8_ig65m(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M model for clips of length 8 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 487, "pretrained on 487 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_8_ig65m",
                       pretrained=pretrained, progress=progress)


def r2plus1d_34_32_ig65m(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M model for clips of length 32 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads weights pretrained on 65 million Instagram videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 359, "pretrained on 359 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_ig65m",
                       pretrained=pretrained, progress=progress)


def r2plus1d_34_8_kinetics(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M-Kinetics model for clips of length 8 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads IG65M weights fine-tuned on Kinetics videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_8_kinetics",
                       pretrained=pretrained, progress=progress)


def r2plus1d_34_32_kinetics(num_classes, pretrained=False, progress=False):
    """R(2+1)D 34-layer IG65M-Kinetics model for clips of length 32 frames.

    Args:
      num_classes: Number of classes in last classification layer
      pretrained: If True, loads IG65M weights fine-tuned on Kinetics videos
      progress: If True, displays a progress bar of the download to stderr
    """
    assert not pretrained or num_classes == 400, "pretrained on 400 classes"
    return r2plus1d_34(num_classes=num_classes, arch="r2plus1d_34_32_kinetics",
                       pretrained=pretrained, progress=progress)


def r2plus1d_34(num_classes, pretrained=False, progress=False, arch=None):
    model = VideoResNet(block=BasicBlock,
                        conv_makers=[Conv2Plus1D] * 4,
                        layers=[3, 4, 6, 3],
                        stem=R2Plus1dStem)

    model.fc = nn.Linear(model.fc.in_features, out_features=num_classes)

    # Fix difference in PyTorch vs Caffe2 architecture
    # https://github.com/facebookresearch/VMZ/issues/89
    # https://github.com/pytorch/vision/issues/1265
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)

    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(model_urls[arch],
                                                        progress=progress)
        model.load_state_dict(state_dict)

    return model
