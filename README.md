# SAL (Neural Networks 2025)
The official implementation of "Multilevel Semantic and Adaptive Actionness Learning for Weakly Supervised Temporal Action Localization".

## The source code will be released after the paper is published.

## Abstract
Weakly supervised temporal action localization aims to identify and localize action instances in untrimmed videos with only video-level labels. Typically, most methods are based on a multiple instance learning framework that uses a top-K strategy to select salient segments to represent the entire video. Therefore fine-grained video information cannot be learned, resulting in poor action classification and localization performance. In this paper, we propose a Multilevel Semantic and Adaptive Actionness Learning Network SAL, which is mainly composed of a multilevel semantic learning MSL branch and an adaptive actionness learning AAL branch. The MSL branch introduces second-order video semantics, which can capture fine-grained information in videos and improve video-level classification performance. Furthermore, we propagate second-order semantics to action segments to enhance the difference between different actions. The AAL branch uses pseudo labels to learn class-agnostic action information. It introduces a video segments mix-up strategy to enhance foreground generalization ability and adds an adaptive actionness mask to balance the quality and quantity of pseudo labels, thereby improving the stability of training. Extensive experiments show that SAL achieves state-of-the-art results on three benchmarks.

## Results
|  Dataset         | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7| AVG(0.1:0.5) | AVG(0.1:0.7) |
| -----------      | --- | --- | ----| ----| ----| ---| -- | ---- | -----|
| THUMOS14         | 76.3| 71.6| 63.7| 54.2| 41.8| 29.0| 17.9| 61.5| 50.6|

|  Dataset         | 0.5 | 0.75 | 0.95 | AVG(0.5:0.95) |
| -----------      | --- | --- | ----| ----|
| ActivityNet 1.2  | 48.5| 31.4| 7.1 |30.8|
| ActivityNet 1.3  |44.5| 28.9| 6.8| 28.8|

## Preparation
CUDA Version: 11.7

Pytorch: 1.12.0

Numpy: 1.23.1 

Python: 3.9.7

GPU: NVIDIA 3090

Dataset: Download the two-stream I3D features for THUMOS'14 to "DATA_PATH". You can download them from [Google Drive](https://drive.google.com/file/d/1paAv3FsqHtNsDO6M78mj7J3WqVf_CgSG/view?usp=sharing).

Update the data_path in "./scripts/train.sh" and "./scripts/inference.sh".

You can download our trained model from [here(Extract code:XXXX)]().

## Training
```
    bash ./scripts/train.sh
```

## Inference
```
    bash ./scripts/inference.sh
```
## Citation
If this work is helpful for your research, please consider citing our works.
