# IG-65M PyTorch

Unofficial PyTorch (and ONNX) 3D video classification models & weights pre-trained on IG-65M (65MM Instagram videos). The official research Caffe2 model and weights are available [here.](https://github.com/facebookresearch/vmz)

## Caffe models :coffee:
Official implementation from FB provide pre-trained models as `.pkl` files. You can fetch them from [here](https://github.com/facebookresearch/VMZ/blob/master/tutorials/model_zoo.md#r21d-34) or the links :point_down: and use our conversion tools.

- [r2plus1d_34_clip8_ig65m_from_scratch](https://www.dropbox.com/s/y8vx3gihhsd8f5b/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl?)
- [r2plus1d_34_clip8_ft_kinetics_from_ig65m](https://www.dropbox.com/s/p81twy88kwrrcop/r2plus1d_34_clip8_ft_kinetics_from_ig65m_%20f128022400.pkl)
- [r2plus1d_34_clip32_ig65m_from_scratch](https://www.dropbox.com/s/eimo232tqw8mwi9/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl)
- [r2plus1d_34_clip32_ft_kinetics_from_ig65m](https://www.dropbox.com/s/z41ff7vs0bzf6b8/r2plus1d_34_clip32_ft_kinetics_from_ig65m_%20f106169681.pkl)

## PyTorch + ONNX Models :trophy:

We also [provide](https://github.com/moabitcoin/ig65m-pytorch/releases) converted `.pth` & `.pb` PyTorch / ONNX weights as artefacts in our Github releases. For models fine-tuned with Kinetics dataset you can use label JSON file included in [here](https://github.com/Showmax/kinetics-downloader/blob/68bd8bc3b9e30da83db9e34cb7d867dcda705cb4/resources/classes.json) with `extract.py` below. 

*Disclaimer*: ONNX models provided here have *NOT* been optimized for inference

| Model  | Pretrain\+Finetune  | Input Size | pth | onnx |
|-------------|:-------------------------|:----------|:-----------------------------------------------|:-------------------------------------------|
|  R(2+1)D_34   | IG-65M + None          |  8x112x112 | [*r2plus1d_34_clip8_ig65m_from_scratch_9bae36ae.pth*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch_9bae36ae.pth)    | [*r2plus1d_34_clip8_ig65m_from_scratch_748ab053.pb*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ig65m_from_scratch_748ab053.pb)     |
|  R(2+1)D_34   | IG-65M + Kinetics  |  8x112x112 | [*r2plus1d_34_clip8_ft_kinetics_from_ig65m_0aa0550b.pth*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m_0aa0550b.pth)  | [*r2plus1d_34_clip8_ft_kinetics_from_ig65m_625d61b3.pb*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip8_ft_kinetics_from_ig65m_625d61b3.pb) |
|  R(2+1)D_34   | IG-65M + None       | 32x112x112 | [*r2plus1d_34_clip32_ig65m_from_scratch_449a7af9.pth*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch_449a7af9.pth)                                               | [*r2plus1d_34_clip32_ig65m_from_scratch_e304d648.pb*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ig65m_from_scratch_e304d648.pb)                                            |
|  R(2+1)D_34   | IG-65M + Kinetics  | 32x112x112 | [*r2plus1d_34_clip32_ft_kinetics_from_ig65m_ade133f1.pth*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m_ade133f1.pth) | [*r2plus1d_34_clip32_ft_kinetics_from_ig65m_10f4c3bf.pb*](https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m_10f4c3bf.pb)  |


### Convert model :spaghetti:
```
python convert.py --help
convert.py [-h] --frames {8,32} --classes {400,487} pkl out

positional arguments:
  pkl                  .pkl file to read the R(2+1)D 34 layer weights from
  out                  prefix to save converted R(2+1)D 34 layer weights to

optional arguments:
  -h, --help           show this help message and exit
  --frames {8,32}      clip frames for video model
  --classes {400,487}  classes in last layer
```

### Extract features :cookie:

```
python extract --help
usage: extract.py [-h] --frames {8,32} --classes {400,487}
                  [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS]
                  [--labels LABELS]
                  model video

positional arguments:
  model                 .pth file to load model weights from
  video                 video file to run feature extraction on

optional arguments:
  -h, --help            show this help message and exit
  --frames {8,32}       clip frames for video model
  --classes {400,487}   classes in last layer
  --batch-size BATCH_SIZE
                        number of sequences per batch for inference
  --num-workers NUM_WORKERS
                        number of workers for data loading
  --labels LABELS       JSON file with label map array
```

## References :book:
1. D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri. **A Closer Look at Spatiotemporal Convolutions for Action Recognition.** CVPR 2018.
2. D. Tran, H. Wang, L. Torresani and M. Feiszli. **Video Classification with Channel-Separated Convolutional Networks.** ICCV 2019.
3. D. Ghadiyaram, M. Feiszli, D. Tran, X. Yan, H. Wang and D. Mahajan, **Large-scale weakly-supervised pre-training for video action recognition.** CVPR 2019.
4. [VMZ: Model Zoo for Video Modeling](https://github.com/facebookresearch/vmz)
5. [Kinetics](https://arxiv.org/abs/1705.06950) & [IG-65M](https://arxiv.org/abs/1905.00561)


## License

Copyright © 2019 MoabitCoin

Distributed under the MIT License (MIT).
