<div align="center">
<!-- <h1>Matte Anything!🐒</h1> -->
<h1> Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation </h3>

<h5 align="center">
    <a href="https://udonda.github.io/">Daichi Horita</a><sup>1</sup>&emsp;
    <a href="https://naoto0804.github.io/">Naoto Inoue</a><sup>2</sup>&emsp;
    <a href="https://ktrk115.github.io/">Kotaro Kikuchi</a><sup>2</sup>&emsp;
    <a href="https://sites.google.com/view/kyamagu">Kota Yamaguchi</a><sup>2</sup>&emsp;
    <a href="https://scholar.google.co.jp/citations?user=CJRhhi0AAAAJ&hl=en">Kiyoharu Aizawa</a><sup>1</sup>&emsp;
    <br>
    <sup>1</sup>The University of Tokyo,
    <sup>2</sup>CyberAgent
</h5>

<h3 align="center">
CVPR 2024
</h3>


[![arxiv paper](https://img.shields.io/badge/arxiv-paper-orange)](https://arxiv.org/abs/2311.13602)
<a href='https://udonda.github.io/RALF/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

`Content-aware graphic layout generation` aims to automatically arrange visual elements along with a given content, such as an e-commerce product image. This repository aims to provide all-in-one package for `content-aware layout generation`. If you like this repository, please give it a star!

![teaser](https://udonda.github.io/RALF/static/images/teaser.png)
In this paper, we propose `Retrieval-augmented content-aware layout generation`. We retrieve nearest neighbor examples based on the input image and use them as a reference to augment the generation process.


## Content
- Setup
- Dataset splits
- Pre-processing Dataset
- Training
- Inference & Evaluation
- Inference using a canvas

## Overview of Benchmark
We provide not only our method (RALF / Autoreg Baseline) but also other state-of-the-art methods for content-aware layout generation. The following methods are included in this repository:
- [Autoreg Baseline \[Horita+ CVPR24\]](https://arxiv.org/abs/2311.13602)
- [RALF \[Horita+ CVPR24\]](https://arxiv.org/abs/2311.13602)
- [CGL-GAN \[Zhou+ IJCAI22\]](https://arxiv.org/abs/2205.00303)
- [DS-GAN \[Hsu+ CVPR23\]](https://arxiv.org/abs/2303.15937)
- [ICVT \[Cao+ ACMMM22\]](https://arxiv.org/abs/2209.00852)
- [LayoutDM \[Inoue+ CVPR23\]](https://arxiv.org/abs/2303.08137)
- [MaskGIT \[Zhang+ CVPR22\]](https://arxiv.org/abs/2202.04200)
- [VQDiffusion \[Gu+ CVPR22\]](https://arxiv.org/abs/2111.14822)

## Setup
We recommend using [Docker](https://www.docker.com/) to easily try our code.


### 1. Requirements
- Python3.9+
- PyTorch 1.13.1

We recommend using Poetry (all settings and dependencies in [pyproject.toml](pyproject.toml)).

### 2. How to install

#### Local environment
1. Install poetry (see [official docs](https://python-poetry.org/docs/)).

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies (it may be slow..)

```bash
poetry install
```

#### Docker environment

1. Build a Docker image
```bash
bash scripts/docker/build.sh
```

2. Attach the container to your shell.
```bash
bash scripts/docker/exec.sh
```

3. Install dependencies in the container

```bash
poetry install
```


### 3. Setup global environment variables
Some variables should be set.　Please make [scripts/bin/setup.sh](https://github.com/CyberAgentAILab/RALF/blob/main/scripts/bin/setup.sh) on your own.　At least these three variables should be set. If you download the provided zip, please ignore the setup.

```bash
DATA_ROOT="./cache/dataset"
```

Some variables might be set (e.g., `OMP_NUM_THREADS`)

### 4. Check Checkpoints and experimental results

The checkpoints and generated layouts of the Autoreg Baseline and our RALF for the unconstrained and constrained tasks are available at [google drive](https://drive.google.com/file/d/1b357gVAnCSqMfbP3Cc2ey6LCeoohfYAi/view?usp=sharing).
After downloading it, please run `unzip cache.zip` in this directory.
Note that the file size is 13GB.

`cache` directory contains:
1. the preprocessed CGL dataset in `cache/dataset`.
2. the weights of the layout encoder and ResNet50 in `cache/PRECOMPUTED_WEIGHT_DIR`.
3. the pre-computed layout feature of CGL in `cache/eval_gt_features`.
4. the relationship of elements for a `relationship` task in `cache/pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt`.
5. the checkpoints and evaluation results of both the Autoreg Baseline and our RALF in `cache/training_logs`.


## Dataset splits

### Train / Test / Val / Real data splits
We perform preprocessing on the PKU and CGL datasets by partitioning the training set into validation and test subsets, as elaborated in Section 4.1.
The CGL dataset, as distributed, is already segmented into these divisions.
For replication of our results, we furnish details of the filenames within the `data_splits/splits/<DATASET_NAME>` directory.
We encourage the use of these predefined splits when conducting experiments based on our setting and using our reported scores such as CGL-GAN and DS-GAN.

### IDs of retrieved samples
We use the training split as a retrieval source. For example, when RALF is trained with the PKU, the training split of PKU is used for training and evaluation.
We provide the pre-computed correspondense using [DreamSim \[Fu+ NeurIPS23\]](https://dreamsim-nights.github.io/) in `data_splits/retrieval/<DATASET_NAME>`. The data structure follows below
```yaml
FILENAME:
    - FILENAME top1
    - FILENAME top2
    ...
    - FILENAME top16
```
You can load an image from `<IMAGE_ROOT>/<FILENAME>.png`.

## Pre-processing Dataset
We highly recommend to pre-process datasets since you can run your experiments as quick as possible!!  
Each script can be used for processing both PKU and CGL by specifying `--dataset_type (pku|cgl)`

### Dataset setup

Folder names with parentheses will be generated by this pipeline.

```
<DATASET_ROOT>
| - annotation
| | (for PKU)
| | - train_csv_9973.csv
| | - test_csv_905.csv
| |  (for CGL)
| | - layout_train_6w_fixed_v2.json
| | - layout_test_6w_fixed_v2.json
| | - yinhe.json
| - image
| | - train
| | | - original: image with layout elements
| | | - (input): image without layout elements (by inpainting)
| | | - (saliency)
| | | - (saliency_sub)
| | - test
| | | - input: image without layout elements
| | | - (saliency)
| | | - (saliency_sub)
```

### Image inpainting

```bash
poetry run python image2layout/hfds_builder/inpainting.py --dataset_root <DATASET_ROOT>
```

### Saliency detection

```bash
poetry run python image2layout/hfds_builder/saliency_detection.py --input_dir <INPUT_DIR> --output_dir <OUTPUT_DIR> (--algorithm (isnet|basnet))
```

### Aggregate data and dump to HFDS
```bash
poetry run python image2layout/hfds_builder/dump_dataset.py --dataset_root <DATASET_ROOT> --output_dir <OUTPUT_DIR>
```

## Training

### Tips
`configs/<METHOD>_<DATASET>.sh` contains the hyperparameters and settings for each method and dataset. Please refer to the file for the details.
In particular, please check whether the debugging mode `DEBUG=True or False`.

### Autoreg Baseline with CGL

Please run
```bash
bash scripts/train/autoreg_cgl.sh <GPU_ID> <TASK_NAME>
# If you wanna run train and eval, please run
bash scripts/run_job/end_to_end.sh <GPU_ID e.g. 0> autoreg cgl <TASK_NAME e.g. uncond>
```
where `TASK_NAME` indicates the unconstrained and constrained tasks.
Please refer to the below task list:  
1. `uncond`: Unconstraint generation  
2. `c`: Category &rarr; Size + Position  
3. `cwh`: Category + Size &rarr; Position  
4. `partial`: Completion
5. `refinement`: Refinement
6. `relation`: Relationship


### RALF with CGL
The dataset with inpainting.

Please run
```bash
bash scripts/train/ralf_cgl.sh <GPU_ID> <TASK_NAME>
# If you wanna run train and eval, please run
bash scripts/run_job/end_to_end.sh <GPU_ID e.g. 0> ralf cgl <TASK_NAME e.g. uncond>
```

### Other methods
For example, these scripts are helpful. `end_to_end.sh` is a wrapper script for training, inference, and evaluation.
```bash
# DS-GAN with CGL dataset
bash scripts/run_job/end_to_end.sh 0 dsgan cgl uncond
# LayoutDM with CGL dataset
bash scripts/run_job/end_to_end.sh 2 layoutdm cgl uncond
# CGL-GAN + Retrieval Augmentation with CGL dataset
bash scripts/run_job/end_to_end.sh 2 cglgan_ra cgl uncond
```


## Inference & Evaluation

Experimental results are provided in `cache/training_logs`. For example, a directory of `autoreg_c_cgl`, which the results of the Autoreg Baseline with Category &rarr; Size + Position task, includes:
1. `test_<SEED>.pkl`: the generated layouts
2. `layout_test_<SEED>.png`: the rendered layouts, in which top sample is ground truth and bottom sample is a predicted sample
3. `gen_final_model.pt`: the final checkpoint
4. `scores_test.tex`: summarized qualitative results

### Annotated split

Please see and run
```bash
bash scripts/eval_inference/eval_inference.sh <GPU_ID> <JOB_DIR> <COND_TYPE> cgl
```


For example,
```bash
# Autoreg Baseline with Unconstraint generation
bash scripts/eval_inference/eval_inference.sh 0 "cache/training_logs/autoreg_uncond_cgl" uncond cgl
```

### Unannotated split
The dataset with real canvas i.e. no inpainting.

Please see and run
```bash
bash scripts/eval_inference/eval_inference_all.sh <GPU_ID>
```




## Inference using a canvas

Please run
```bash
bash scripts/run_job/inference_single_data.sh <GPU_ID> <JOB_DIR> cgl <SAMPLE_ID>
```
where `SAMPLE_ID` can optionally be set as a dataset index. 


For example, 
```bash
bash scripts/run_job/inference_single_data.sh 0 "./cache/training_logs/ralf_uncond_cgl" cgl
```

## Inference using your personal data

Please customize [image2layout/train/inference_single_data.py](https://github.com/CyberAgentAILab/RALF/blob/main/image2layout/train/inference_single_data.py) to load your data.


## Citation
If you find our work useful in your research, please consider citing:
```bibtex
@article{horita2024retrievalaugmented,
    title={{Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation}},
    author={Daichi Horita and Naoto Inoue and Kotaro Kikuchi and Kota Yamaguchi and Kiyoharu Aizawa},
    booktitle={CVPR},
    year={2024}
}
```
