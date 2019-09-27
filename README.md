# IG65-M PyTorch

Unofficial PyTorch (and ONNX) models and weights for IG65-M pre-trained 3d video architectures. The official research Caffe2 model and weights are availabe [here](https://github.com/facebookresearch/vmz)

## Caffe models :coffee:
Official implementation from FB provide pre-trained models as .pkl files. You can fetch them from [here](https://github.com/facebookresearch/VMZ/blob/master/tutorials/model_zoo.md#r21d-34) or links :point_down: and use our conversion tools.

1. [r2plus1d_34_clip8_ig65m_from_scratch](https://www.dropbox.com/s/y8vx3gihhsd8f5b/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl?dl=0)
2. [r2plus1d_34_clip8_ft_kinetics_from_ig65m](https://www.dropbox.com/s/p81twy88kwrrcop/r2plus1d_34_clip8_ft_kinetics_from_ig65m_%20f128022400.pkl?dl=0)
3. [r2plus1d_34_clip32_ig65m_from_scratch](https://www.dropbox.com/s/y8vx3gihhsd8f5b/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl?dl=0)
4. [r2plus1d_34_clip32_ft_kinetics_from_ig65m](https://www.dropbox.com/s/z41ff7vs0bzf6b8/r2plus1d_34_clip32_ft_kinetics_from_ig65m_%20f106169681.pkl?dl=0)

## PyTorch + ONNX Models :trophy:

We also [provide](https://github.com/moabitcoin/ig65m-pytorch/releases) converted `.pth` & `.pb` PyTorch/ONNX weights as artefacts in our Github releases.

*Disclaimer* : ONNX models provided here have *NOT* been optimized for inference

| Model | Weights            | Input Size | pth                                             | onnx                                          |
|:-------------|:------------------|:----------|:-----------------------------------------------|:---------------------------------------------|
|  R(2+1)D 34   | IG65-M             |  8x112x112 | *r2plus1d_34_clip8_ig65m_from_scratch-9bae36ae.pth*      | *r2plus1d_34_clip8_ig65m_from_scratch-748ab053.pb*     |
|  R(2+1)D 34   | IG65-M + Kinetics  |  8x112x112 | *r2plus1d_34_clip8_ft_kinetics_from_ig65m-0aa0550b.pth*  | *r2plus1d_34_clip8_ft_kinetics_from_ig65m-625d61b3.pb* |
|  R(2+1)D 34   | IG65-M             | 32x112x112 | NA                                              | NA                                            |
|  R(2+1)D 34   | IG65-M + Kinetics  | 32x112x112 | *r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth* | r2plus1d_34_clip32_ft_kinetics_from_ig65m-10f4c3bf.pb  |


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


## References
- [VMZ: Model Zoo for Video Modeling](https://github.com/facebookresearch/vmz)
- [Kinetics dataset paper on arxiv](https://arxiv.org/abs/1705.06950)
- [IG-65M dataset paper on arxiv](https://arxiv.org/abs/1905.00561)


## License

Copyright © 2019 MoabitCoin

Distributed under the MIT License (MIT).
