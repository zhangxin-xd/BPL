# Block-wise Partner Learning for Model Compression
Official Pytorch implementation of [Block-wise Partner Learning for Model Compression](https://ieeexplore.ieee.org/abstract/document/10237122)[Accepted by IEEE TNNLS 2023]
## Getting Started

Download the repo:

```bash
git clone https://github.com/zhangxin-xd/BPL.git
cd BPL
```
Set up the environment:

```bash
conda create -n BPL python=3.8
conda activate BPL
pip install -r requirements.txt
```
Data Preparation

- Put [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data to `~/data_cifar`.
- Put unzipped [ImageNet](https://www.image-net.org/) data to `~/data_imagenet`.

## Citation
```
@ARTICLE{10237122,
  author={Zhang, Xin and Xie, Weiying and Li, Yunsong and Lei, Jie and Jiang, Kai and Fang, Leyuan and Du, Qian},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Block-Wise Partner Learning for Model Compression}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2023.3306512}}
```


