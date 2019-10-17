import sys
import pickle

import torch
import torch.nn as nn

from torchvision.models.video.resnet import BasicBlock, R2Plus1dStem, Conv2Plus1D

from ig65m.models import r2plus1d_34


def blobs_from_pkl(path):
    with path.open(mode="rb") as f:
        pkl = pickle.load(f, encoding="latin1")
        blobs = pkl["blobs"]

        return blobs


def copy_tensor(data, blobs, name):
    tensor = torch.from_numpy(blobs[name])

    del blobs[name]  # enforce: use at most once

    assert data.size() == tensor.size()
    assert data.dtype == tensor.dtype

    data.copy_(tensor)


def copy_conv(module, blobs, prefix):
    assert isinstance(module, nn.Conv3d)
    assert module.bias is None
    copy_tensor(module.weight.data, blobs, prefix + "_w")


def copy_bn(module, blobs, prefix):
    assert isinstance(module, nn.BatchNorm3d)
    copy_tensor(module.weight.data, blobs, prefix + "_s")
    copy_tensor(module.running_mean.data, blobs, prefix + "_rm")
    copy_tensor(module.running_var.data, blobs, prefix + "_riv")
    copy_tensor(module.bias.data, blobs, prefix + "_b")


def copy_fc(module, blobs):
    assert isinstance(module, nn.Linear)
    n = module.out_features
    copy_tensor(module.bias.data, blobs, "last_out_L" + str(n) + "_b")
    copy_tensor(module.weight.data, blobs, "last_out_L" + str(n) + "_w")


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L174-L188
# https://github.com/facebookresearch/VMZ/blob/6c925c47b7d6545b64094a083f111258b37cbeca/lib/models/r3d_model.py#L233-L275
def copy_stem(module, blobs):
    assert isinstance(module, R2Plus1dStem)
    assert len(module) == 6
    copy_conv(module[0], blobs, "conv1_middle")
    copy_bn(module[1], blobs, "conv1_middle_spatbn_relu")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "conv1")
    copy_bn(module[4], blobs, "conv1_spatbn_relu")
    assert isinstance(module[5], nn.ReLU)


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_conv2plus1d(module, blobs, i, j):
    assert isinstance(module, Conv2Plus1D)
    assert len(module) == 4
    copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j) + "_middle")
    copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j) + "_middle")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "comp_" + str(i) + "_conv_" + str(j))


# https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/video/resnet.py#L82-L114
def copy_basicblock(module, blobs, i):
    assert isinstance(module, BasicBlock)

    assert len(module.conv1) == 3
    assert isinstance(module.conv1[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv1[0], blobs, i, 1)
    assert isinstance(module.conv1[1], nn.BatchNorm3d)
    copy_bn(module.conv1[1], blobs, "comp_" + str(i) + "_spatbn_" + str(1))
    assert isinstance(module.conv1[2], nn.ReLU)

    assert len(module.conv2) == 2
    assert isinstance(module.conv2[0], Conv2Plus1D)
    copy_conv2plus1d(module.conv2[0], blobs, i, 2)
    assert isinstance(module.conv2[1], nn.BatchNorm3d)
    copy_bn(module.conv2[1], blobs, "comp_" + str(i) + "_spatbn_" + str(2))

    if module.downsample is not None:
        assert i in [3, 7, 13]
        assert len(module.downsample) == 2
        assert isinstance(module.downsample[0], nn.Conv3d)
        assert isinstance(module.downsample[1], nn.BatchNorm3d)
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn")


def copy_layer(module, blobs, i):
    assert {0: 3, 3: 4, 7: 6, 13: 3}[i] == len(module)

    for basicblock in module:
        copy_basicblock(basicblock, blobs, i)
        i += 1


def init_canary(model):
    nan = float("nan")

    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            nn.init.constant_(m.weight, nan)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.running_mean, nan)
            nn.init.constant_(m.running_var, nan)
            nn.init.constant_(m.bias, nan)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.bias, nan)


def check_canary(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            assert not torch.isnan(m.weight).any()
        elif isinstance(m, nn.BatchNorm3d):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.running_mean).any()
            assert not torch.isnan(m.running_var).any()
            assert not torch.isnan(m.bias).any()
        elif isinstance(m, nn.Linear):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.bias).any()


def main(args):
    blobs = blobs_from_pkl(args.pkl)

    if not "last_out_L{}_w".format(args.classes) in blobs:
        sys.exit("Error: number of --classes does not match the last linear layer in .pkl blobs")

    if not "last_out_L{}_b".format(args.classes) in blobs:
        sys.exit("Error: number of --classes does not match the last linear layer in .pkl blobs")

    model = r2plus1d_34(num_classes=args.classes)

    init_canary(model)

    copy_stem(model.stem, blobs)

    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    blocks = [0, 3, 7, 13]

    for layer, i in zip(layers, blocks):
        copy_layer(layer, blobs, i)

    copy_fc(model.fc, blobs)

    assert not blobs
    check_canary(model)

    # Export to pytorch .pth and self-contained onnx .pb files

    batch = torch.rand(1, 3, args.frames, 112, 112)  # NxCxTxHxW
    torch.save(model.state_dict(), args.out.with_suffix(".pth"))
    torch.onnx.export(model, batch, args.out.with_suffix(".pb"))

    # Check pth roundtrip into fresh model

    model = r2plus1d_34(num_classes=args.classes)
    model.load_state_dict(torch.load(args.out.with_suffix(".pth")))
