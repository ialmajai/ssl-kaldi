# Lipreading Recipe using AV-Hubert and Grid Corpus


## AV-HuBERT Setup

```
conda activate ssl-kaldi
pip install pip==24.0
pip install numpy==1.19.5
pip install -r requirements.txt
git clone https://github.com/facebookresearch/av_hubert.git --depth=1
cd av_hubert
git submodule init
git submodule update
cd fairseq
pip install --editable ./
conda install -c conda-forge dlib==19.18.0
```
## Unseen-speaker performance

| System    | Setting    | Grammar/LM   | Data augmentation |   WER (unseen speaker) |
| ------- | ------- | ------- | ------- | ------- |
| LipNet Grid | unseen speakers | Fixed GRID grammar    | Spatial/temporal jitter, mirroring, etc.  | 11.4% |<200b>
| AV‑HuBERT Grid e2e | unseen speakers |    Fixed GRID grammar |    None |  7.15% |



## Citation

If you use this recipe, please cite **ssl-kaldi** and the AV‑HuBERT paper
```
@misc{ssl_kaldi,
author = {Ibrahim Almajai},
title = {ssl-kaldi: SSL features are all you need,
year = {2025},
howpublished = {\url{https://github.com/ialmajai/ssl-kaldi}}
note = {Accessed: 2025-12}
}

@article{shi2022avhubert,
    author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
    title = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
    journal = {arXiv preprint arXiv:2201.02184}
    year = {2022}
}
```
Contact
Author: Ibrahim Almajai (ialmajai@gmail.com)

