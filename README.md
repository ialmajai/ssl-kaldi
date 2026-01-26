# ssl-kaldi 
Self-Supervised Learning Features for [Kaldi](https://github.com/kaldi-asr/kaldi) ASR

This repository contains recipes and tools for integrating self-supervised learning (SSL) features such as HuBERT, mHuBERT, and AV-HuBERT into Kaldi ASR systems. It bridges modern SSL models with Kaldi's robust pipelines, enabling efficient feature extraction, dimensionality reduction, and end-to-end training for low-resource and standard datasets. The approach avoids the need for fine-tuning large pretrained models.

## Pipeline Overview
```
Audio / Video
      ↓
Pretrained SSL model (PyTorch)
      ↓
Frame-level feature extraction
      ↓
PCA dimensionality reduction / Upsampling (Optional)
      ↓
Kaldi ark/scp features
      ↓
Standard Kaldi training & decoding
```

### Prerequisites
Kaldi Installation: Follow official Kaldi setup or use Docker image.

Suggestions for improvements or new features are always welcome! Feel free to open an issue or submit a pull request.

## Getting Started:

### Create a conda environment
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
