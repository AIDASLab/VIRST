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
- [ ] release eval script
- [ ] release training scripts
- [ ] demo script

## Overview

This repository contains the core training and evaluation code for VIRST, including:

- model definition in `model/`
- training entrypoints in `train.py` and `train_stage3.py`
- RVOS evaluation in `eval.py`
- dataset handling in `data/`
- utility code in `utils/`

## Checkpoint

Pretrained checkpoint: [Google Drive](https://drive.google.com/file/d/19PrTMWWzGHBTrZ0JTe1feH205vjHkoNx/view?usp=sharing)

## Notes

- The project page will be updated as the release is polished further.

## Acknowledgements

This project builds upon prior work, including 
[VISA](https://github.com/cilinyan/VISA), 
[LISA](https://github.com/JIA-Lab-research/LISA), 
[VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash), 
and [SAM2](https://github.com/facebookresearch/sam2).

We thank the authors for releasing their code and models.