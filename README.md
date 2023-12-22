# Block-wise Partner Learning for Model Compression
Official Pytorch implementation of [Block-wise Partner Learning for Model Compression](https://ieeexplore.ieee.org/abstract/document/10237122) [Accepted by IEEE TNNLS 2023]
## Getting Started

Download the repo:

```bash
git clone https://github.com/zhangxin-xd/BPL.git
cd BPL
```

Data Preparation

- Put [NWPU-45](https://www.kaggle.com/datasets/happyyang/nwpu-data-set) data to `~/data_nwpu`.
- Put [UCML-21](http://weegee.vision.ucmerced.edu/datasets/landuse.html) data to `~/data_ucml`.

Pruning 

```
CUDA_VISIBLE_DEVICES=0 python main.py --dataset nwpu-45 --data_path ~/data_nwpu --block_type shadow --pruning_rate 0.5
```

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


