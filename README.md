# IG65-M PyTorch

Unofficial PyTorch (and ONNX) models and weights for IG65-M pre-trained 3d video architectures.

The official research Caffe2 model and weights are availabe at: https://github.com/facebookresearch/vmz


## Models

| Model         | Weights            | Input Size | pth                                             | onnx                                          |
| ------------- | ------------------ | ---------- | ----------------------------------------------- | --------------------------------------------- |
|  r(2+1)d 34   | IG65-M             |  8x112x112 | *r2plus1d_34_clip8_ig65m_from_scratch.pth*      | *r2plus1d_34_clip8_ig65m_from_scratch.pb*     |
|  r(2+1)d 34   | IG65-M + Kinetics  |  8x112x112 | *r2plus1d_34_clip8_ft_kinetics_from_ig65m.pth*  | *r2plus1d_34_clip8_ft_kinetics_from_ig65m.pb* |
|  r(2+1)d 34   | IG65-M             | 32x112x112 | NA                                              | NA                                            |
|  r(2+1)d 34   | IG65-M + Kinetics  | 32x112x112 | *r2plus1d_34_clip32_ft_kinetics_from_ig65m.pth* | r2plus1d_34_clip32_ft_kinetics_from_ig65m.pb  |


## Usage

See
- `convert.py` for model conversion
- `extract.py` for feature extraction

We provide converted `.pth` PyTorch weights as artifacts in our Github releases.


## References
- [VMZ: Model Zoo for Video Modeling](https://github.com/facebookresearch/vmz)
- [Kinetics](https://arxiv.org/abs/1705.06950)
- [IG65-M](https://arxiv.org/abs/1905.00561)


## License

Copyright © 2019 MoabitCoin

Distributed under the MIT License (MIT).
