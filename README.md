# Cross-Domain Few-Shot 3D Point Cloud Semantic Segmentation [[pdf](https://www.sciencedirect.com/science/article/abs/pii/S0167865525002533)]

## Running 
**Installation and data preparation please follow [attMPTI](https://github.com/Na-Z/attMPTI).**

### Training

Pretrain the segmentor which includes feature extractor module on the available training set:

```bash
bash scripts/pretrain_segmentor.sh
```

```
Train our method  under cross domain few-shot setting:

```bash
bash scripts/train_PAP.sh
```

### Evaluation

Test our method under cross domain few-shot setting:

```bash
bash scripts/eval_PAP.sh
```


## Citation
Please cite our paper if it is helpful to your research:

 @article{xiao2025cross,
  title={Cross-Domain Few-Shot 3D Point Cloud Semantic Segmentation},
  author={Xiao, Jiwei and Wang, Ruiping and He, Chen and Chen, Xilin},
  journal={Pattern Recognition Letters},
  year={2025},
  publisher={Elsevier}
}


## Acknowledgement
We thank [DGCNN (pytorch)](https://github.com/WangYueFt/dgcnn/tree/master/pytorch) and [attMPTI](https://github.com/Na-Z/attMPTI) for sharing their source code.
