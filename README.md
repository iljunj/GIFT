# GIFT
The implementation of our ACM MM 2024 paper "Transferable Adversarial Facial Images for Privacy Protection"

![Python 3.8.18](https://img.shields.io/badge/python-3.8.18-green.svg?style=plastic)
![Pytorch 1.12.1](https://img.shields.io/badge/pytorch-1.12.1-red.svg?style=plastic)

## Abstract
The success of deep face recognition (FR) systems has raised serious privacy concerns due to their ability to enable unauthorized tracking of users in the digital world. Previous studies proposed introducing imperceptible adversarial noises into face images to deceive those face recognition models, thus achieving the goal of enhancing facial privacy protection. Nevertheless, they heavily rely on user-chosen references to guide the generation of adversarial noises, and cannot simultaneously construct natural and highly transferable adversarial face images in black-box scenarios. In light of this, we present a novel face privacy protection scheme with improved transferability while maintain high visual quality. We propose shaping the entire face space directly instead of exploiting one kind of facial characteristic like makeup information to integrate adversarial noises. To achieve this goal, we first exploit global adversarial latent search to traverse the latent space of the generative model, thereby creating natural adversarial face images with high transferability. We then introduce a key landmark regularization module to preserve the visual identity information. Finally, we investigate the impacts of various kinds of latent spaces and find that F latent space benefits the trade-off between visual naturalness and adversarial transferability. Extensive experiments over two datasets demonstrate that our approach significantly enhances attack transferability while maintaining high visual quality, outperforming state-of-the-art methods by an average 25% improvement in deep FR models and 10% improvement on commercial FR APIs.

## Latest Update
**2024/12/2** We have released the official implementation code.

## Setup
- **Get code**
```shell
git clone git@github.com:iljunj/GIFT.git
```
- **Build environment**
```shell
cd GIFT
conda env create -f environment.yml
```
- **Download FR models and datasets**
  - Pretrained face recognition models and the subset of a subset of CelebA-HQ for evaluation offered by [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN):
    [[Google](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view?usp=sharing)] [[Baidu](https://pan.baidu.com/s/1hiIV1GVZTwV1o2Q4DfC2Cg)] pw:1bpv
  - put the subset of CelebA-HQ and Pretrained face recognition models in ```GIFT/GIFTInvert/```
- **Download pretrained base code encoder**
  Download the pretrained base code encoder offered by [BDInvert](https://github.com/kkang831/BDInvert_Release) and unzip under `GIFT/GIFTInvert/pretrained_models/`.
  | Encoder Pretrained Models                   | Basc Code Spatial Size |
  | :--                                         | :--    |
  | [StyleGAN2 pretrained on FFHQ 1024, 16x16](https://drive.google.com/file/d/1Gwi7I72vL7rdwET1Q0QnR71ZuZ0M3Jx1/view?usp=sharing)    | 16x16
- **Download the pre-trained semantic encoder**
  unzip the folder under `GIFT/GIFTInvert/` and  download the pre-trained model [our pre-trained model](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) offered by [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch) and save it in `GIFT/GIFTInvert/faceparsing/res/cp`.
- **The final project should be like this:**
    ```shell
    GIFT
    └- GIFTInvert
       └- CelebA-HQ
       └- face_models
       └- faceparsing
          └- res/cp/checkpoints
       └- pretrained_models
       └- invert.py
       └- adv_facenet.py
       └- make_list.py
       └- ...
    ```

## Quick Start
1. Change directory into BDInvert.
```shell
cd GIFT/GIFTInvert
```
2. Make image list.
```shell
python make_list.py
```
3. Latent Code Initialization.
```shell
python invert.py
```
4.search adversarial highly transferable adversarial example
```shell
python adv_facenet.py
```

## Acknowledge
Some of our code are based on [BDInvert](https://github.com/kkang831/BDInvert_Release)，[AMT-GAN](https://github.com/CGCL-codes/AMT-GAN) and [BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch).

## BibTeX 
If you find GIFT both interesting and helpful, please consider citing us in your research or publications:
```bibtex
@inproceedings{li2024transferable,
  title={Transferable Adversarial Facial Images for Privacy Protection},
  author={Li, Minghui and Wang, Jiangxiong and Zhang, Hao and Zhou, Ziqi and Hu, Shengshan and Pei, Xiaobing},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={10649--10658},
  year={2024}
}
```



