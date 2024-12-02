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

- **Build environment**
```shell
cd GIFT
conda env create -f environment.yml
```

## Quick Start

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



