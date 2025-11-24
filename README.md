# ssl-kaldi 
Self-Supervised Learning Features in Kaldi

## SSL features are all you need!

This repository contains recipes and tools for integrating self-supervised learning (SSL) features into Kaldi ASR systems. It bridges modern SSL models with Kaldi's robust pipelines, enabling efficient feature extraction, dimensionality reduction, and end-to-end training for low-resource and standard datasets.


PCA Dimensionality Reduction: For SSL features in GMM-HMM systems.

Kaldi Compatibility: Features are extracted in Kaldi's standard ark and scp file formats, ensuring seamless integration with Kaldi pipelines.

Prerequisites
Kaldi Installation: Follow official Kaldi setup or use Docker image.

Hardware: GPU recommended for SSL feature extraction including PCA.

Getting Started

# Create environment
```
git clone https://github.com/ialmajai/ssl-kaldi.git
cd ssl-kaldi
conda create -n ssl-kaldi python=3.8 -y
conda activate ssl-kaldi

pip install -r requirements.txt
```

## Citation
```
@misc{ssl_kaldi,
author = {Ibrahim Almajai},
title = {ssl-kaldi: SSL features are all you need,
year = {2025},
howpublished = {\url{https://github.com/ialmajai/ssl-kaldi}}
note = {Accessed: 2025-11}
}

Contact
Author: Ibrahim Almajai (ialmajai@gmail.com)
