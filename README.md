## Minimal Generative Model Evaluation Harness

This is the evaluation harness extracted from the PyTorch implementation of [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch), extended to be applicable to generative models beyond StyleGAN-esque models.
## Requirements

* Linux and Windows are supported, but we recommend Linux for performance and compatibility reasons.
* 1&ndash;8 high-end NVIDIA GPUs with at least 2 GB of memory. We have done all testing and development using NVIDIA DGX-1 with 8 Tesla V100 GPUs.
* 64-bit Python 3.7 and PyTorch 1.7.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later.  Use at least version 11.1 if running on RTX 3090.  (Why is a separate CUDA toolkit installation required?  See comments in [#2](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/2#issuecomment-779457121).)
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3`.  We use the Anaconda3 2020.11 distribution which installs most of these by default.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

The code relies heavily on custom PyTorch extensions that are compiled on the fly using NVCC. On Windows, the compilation requires Microsoft Visual Studio. We recommend installing [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/) and adding it into `PATH` using `"C:\Program Files (x86)\Microsoft Visual Studio\<VERSION>\Community\VC\Auxiliary\Build\vcvars64.bat"`.

## Getting started

To evaluate your generative model, use an arbitrary method to form a folder of certain generated samples. Then, prepare the reference dataset and the test set by instructions to run `dataset_tool.py` explained below. Finally, run `calc_metrics.py` to evaluate common metrics:

```.python
python3 calc_metrics.py --metrics=kid50k_full --data=cifar10.zip --testdata={TEST_DATA}.zip
```

**Docker**: You can run the above curated image example using Docker as follows:

```.bash
docker build --tag sg2ada:latest .
./docker_run.sh python3 calc_metrics.py --metrics=kid50k_full --data=./cifar10.zip --testdata=./cifar10.zip --resolution=32
```

Note: The Docker image requires NVIDIA driver release `r455.23` or later.

## Preparing datasets

Datasets are stored as uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` for labels.

Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) for more information. Alternatively, the folder can also be used directly as a dataset, without running it through `dataset_tool.py` first, but doing so may lead to suboptimal performance.

Legacy TFRecords datasets are not supported &mdash; see below for instructions on how to convert them.

**FFHQ**:

Step 1: Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as TFRecords.

Step 2: Extract images from TFRecords using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python ../stylegan2-ada/dataset_tool.py unpack \
    --tfrecord_dir=~/ffhq-dataset/tfrecords/ffhq --output_dir=/tmp/ffhq-unpacked
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
# Original 1024x1024 resolution.
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq.zip

# Scaled down 256x256 resolution.
python dataset_tool.py --source=/tmp/ffhq-unpacked --dest=~/datasets/ffhq256x256.zip \
    --width=256 --height=256
```

**MetFaces**: Download the [MetFaces dataset](https://github.com/NVlabs/metfaces-dataset) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/metfaces/images --dest=~/datasets/metfaces.zip
```

**AFHQ**: Download the [AFHQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) and create ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/afhq/train/cat --dest=~/datasets/afhqcat.zip
python dataset_tool.py --source=~/downloads/afhq/train/dog --dest=~/datasets/afhqdog.zip
python dataset_tool.py --source=~/downloads/afhq/train/wild --dest=~/datasets/afhqwild.zip
```

**CIFAR-10**: Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/cifar-10-python.tar.gz --dest=~/datasets/cifar10.zip
```

**LSUN**: Download the desired categories from the [LSUN project page](https://www.yf.io/p/lsun/) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=~/downloads/lsun/raw/cat_lmdb --dest=~/datasets/lsuncat200k.zip \
    --transform=center-crop --width=256 --height=256 --max_images=200000

python dataset_tool.py --source=~/downloads/lsun/raw/car_lmdb --dest=~/datasets/lsuncar200k.zip \
    --transform=center-crop-wide --width=512 --height=384 --max_images=200000
```

**BreCaHAD**:

Step 1: Download the [BreCaHAD dataset](https://figshare.com/articles/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186).

Step 2: Extract 512x512 resolution crops using `dataset_tool.py` from the [TensorFlow version of StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada/):

```.bash
# Using dataset_tool.py from TensorFlow version at
# https://github.com/NVlabs/stylegan2-ada/
python dataset_tool.py extract_brecahad_crops --cropsize=512 \
    --output_dir=/tmp/brecahad-crops --brecahad_dir=~/downloads/brecahad/images
```

Step 3: Create ZIP archive using `dataset_tool.py` from this repository:

```.bash
python dataset_tool.py --source=/tmp/brecahad-crops --dest=~/datasets/brecahad.zip
```

## Metrics

We employ the following metrics in the ADA paper. Execution time and GPU memory usage is reported for one NVIDIA Tesla V100 GPU at 1024x1024 resolution:

| Metric        | Time   | GPU mem | Description |
| :-----        | :----: | :-----: | :---------- |
| `fid50k_full` | 13 min | 1.8 GB  | Fr&eacute;chet inception distance<sup>[1]</sup> against the full dataset
| `kid50k_full` | 13 min | 1.8 GB  | Kernel inception distance<sup>[2]</sup> against the full dataset
| `pr50k3_full` | 13 min | 4.1 GB  | Precision and recall<sup>[3]</sup> againt the full dataset
| `is50k`       | 13 min | 1.8 GB  | Inception score<sup>[4]</sup> for CIFAR-10

In addition, the following metrics from the [StyleGAN](https://github.com/NVlabs/stylegan) and [StyleGAN2](https://github.com/NVlabs/stylegan2) papers are also supported:

| Metric        | Time   | GPU mem | Description |
| :------------ | :----: | :-----: | :---------- |
| `fid50k`      | 13 min | 1.8 GB  | Fr&eacute;chet inception distance against 50k real images
| `kid50k`      | 13 min | 1.8 GB  | Kernel inception distance against 50k real images
| `pr50k3`      | 13 min | 4.1 GB  | Precision and recall against 50k real images
| `ppl2_wend`   | 36 min | 2.4 GB  | Perceptual path length<sup>[5]</sup> in W, endpoints, full image
| `ppl_zfull`   | 36 min | 2.4 GB  | Perceptual path length in Z, full paths, cropped image
| `ppl_wfull`   | 36 min | 2.4 GB  | Perceptual path length in W, full paths, cropped image
| `ppl_zend`    | 36 min | 2.4 GB  | Perceptual path length in Z, endpoints, cropped image
| `ppl_wend`    | 36 min | 2.4 GB  | Perceptual path length in W, endpoints, cropped image

References:
1. [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500), Heusel et al. 2017
2. [Demystifying MMD GANs](https://arxiv.org/abs/1801.01401), Bi&nacute;kowski et al. 2018
3. [Improved Precision and Recall Metric for Assessing Generative Models](https://arxiv.org/abs/1904.06991), Kynk&auml;&auml;nniemi et al. 2019
4. [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498), Salimans et al. 2016
5. [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948), Karras et al. 2018

## License

Copyright &copy; 2021, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html).

The modification is released under an MIT license.

## Citation

You should probably cite the original authors of the ADA paper:
```
@inproceedings{Karras2020ada,
  title     = {Training Generative Adversarial Networks with Limited Data},
  author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
  booktitle = {Proc. NeurIPS},
  year      = {2020}
}
```

Or, additionally, you may cite [ArtBench](https://github.com/liaopeiyuan/artbench), which uses this harness to evaluate results:

```
@misc{artbench,
  author = {Peiyuan Liao and Xiuyu Li and Xihui Liu and Kurt Keutzer},
  title  = {The ArtBench Dataset: Benchmarking Generative Models with Artworks},
  year   = {2022},
  url    = {https://github.com/liaopeiyuan/artbench}
}
```