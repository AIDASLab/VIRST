<h1 align="center">VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation</h1>
<h3 align="center">CVPR 2026</h3>

<p align="center">
  <a href="static/arch_cr_v1.pdf">
    <img src="static/arch_cr_v1.png" alt="VIRST architecture figure" width="100%">
  </a>
</p>

<p align="center">
  Official implementation of <strong>VIRST</strong>, a video-instructed reasoning framework for spatiotemporal segmentation.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.27060">
    <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
  </a>
  <a href="https://aidaslab.github.io/VIRST/">
    <img src="https://img.shields.io/badge/Project-Page-4285F4.svg">
  </a>
  <a href="https://github.com/AIDASLab/VIRST">
    <img src="https://img.shields.io/badge/GitHub-Code-black.svg">
  </a>
</p>

## TODO

- [x] release model code
- [x] release checkpoint
- [x] release data code
- [x] release utility scripts
- [x] release eval script
- [ ] release training scripts
- [ ] demo script

## Overview

This repository contains the core training and evaluation code for VIRST, including:

- model definition in `model/`
- training entrypoints in `train.py` and `train_stage3.py`
- RVOS evaluation in `eval.py`
- dataset handling in `data/`
- utility code in `utils/`

# Installation 
```
git clone https://github.com/AIDASLab/VIRST
cd VIRST
conda create -n virst python=3.10 -y 
conda activate virst
pip install -r requirements.txt
```


## Checkpoint

Pretrained checkpoint: [Google Drive](https://drive.google.com/file/d/19PrTMWWzGHBTrZ0JTe1feH205vjHkoNx/view?usp=sharing)


## Dataset 
- Download Ref-DAVIS, Ref-YouTube-VOS, [MeViS](https://github.com/henghuiding/MeViS), [ReVOS](https://github.com/cilinyan/VISA)
- By default, `data/dataset_config.py` resolves dataset paths to absolute paths under `<repo>/dataset/`.
- You can override the defaults with `VIRST_LISA_ROOT`, `VIRST_RVOS_ROOT`, `VIRST_CHATUNIVI_ROOT`, and `VIRST_VQA_VIDEO_ROOT`.
- Store them in the following directory 

```
RVOS_ROOT
├── ReVOS
│   ├── JPEGImages 
│   ├── mask_dict.json             
│   ├── mask_dict_foreground.json   
│   ├── meta_expressions_train_.json 
│   └── meta_expressions_valid_.json 
├── lvvis
│   └── train
|       ├── JPEGImages
|       ├── mask_dict.json
|       └── meta_expressions.json
├── Ref-Youtube-VOS
│   ├── meta_expressions
|   |   ├── train/meta_expressions.json
|   |   └── valid/meta_expressions.json
│   ├── train
|   |   ├── JPEGImages
|   |   └── mask_dict.pkl
│   └── valid
|       └── JPEGImages
├── davis17
│   ├── meta_expressions
|   |   ├── train/meta_expressions.json
|   |   └── valid/meta_expressions.json
│   ├── train
|   |   ├── JPEGImages
|   |   └── mask_dict.pkl
│   └── valid
|       ├── JPEGImages
|       └── mask_dict.pkl
└── mevis
```

## Evaluation
To be added 


## Notes

- The project page will be updated as the release is polished further.

## Acknowledgements

This project builds upon prior work, including 
[VISA](https://github.com/cilinyan/VISA), 
[LISA](https://github.com/JIA-Lab-research/LISA), 
[VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash), 
and [SAM2](https://github.com/facebookresearch/sam2).

We thank the authors for releasing their code and models.
