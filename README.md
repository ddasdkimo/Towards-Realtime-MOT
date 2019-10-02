# Towards-Realtime-MOT
**NOTE:** Still in progress, will update constantly, thank you for your attention!

## Introduction
This repo is the a codebase of the Joint Detection and Embedding (JDE) model. JDE is a fast and high-performance multiple-object tracker that learns the object detection task and appearance embedding task simutaneously in a shared neural network. Techical details are described in our [arXiv preprint paper](https://arxiv.org). By using this repo, you can simply achieve **MOTA 64%+** on the "private" protocol of [MOT-16 challenge](https://motchallenge.net/tracker/JDE), and with a near real-time speed at **18~24 FPS** (Note this speed is for the entire system, including the detection step! ) .

We hope this repo will help researches/engineers to develop more practical MOT systems. For algorithm development, we provide training data, baseline models and evaluation methods to make a level playground. For application usage, we also provide a small video demo that takes raw videos as input without any bells and whistles.

## Requirements
* Python 3.6
* [Pytorch](https://pytorch.org) >= 1.0.1
* [syncbn](https://github.com/ytoon/Synchronized-BatchNorm-PyTorch) (Optional, compile and place it under utils/syncbn, or simply replace with nn.BatchNorm [here](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/models.py#L12))
* [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) (Their GPU NMS is used in this project)
* python-opencv
* ffmpeg (Optional, used in the video demo)

## Video Demo

## Dataset zoo

## Pretrained Models

## Test on MOT-16 Challenge

## Training

## Train with custom datasets

## Acknowledgement
A large portion of code is borrowed from [ultralytics/yolov3](https://github.com/ultralytics/yolov3) and [longcw/MOTDT](https://github.com/longcw/MOTDT), many thanks to their wonderful work!