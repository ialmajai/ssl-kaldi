# Lipreading Recipe using AV-Hubert and Grid Corpus


## AV-HuBERT Setup

```
conda activate ssl-kaldi
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
| LipNet | unseen speakers | Fixed GRID grammar    | Spatial/temporal jitter, mirroring, etc.  | 11.4% |<200b>
| SSL-Kaldi GMM-HMM | unseen speakers |    Fixed GRID grammar |    None |  6.36% |

## More Results (WER)
- 30-dimensional PCA for HMM-GMM systems
- Features usampled to 50fps give better results 
- Layer 9 and layer 12 are selected for base and large AV-HuBert
- Speaker adaption (MLLT+SAT) outperform TDNN systems trained on raw features. The power of such adaptation for visual speech features has already been demonstrated in this [paper](https://ueaeprints.uea.ac.uk/id/eprint/63479/1/Accepted_manuscript.pdf)  

| AV-HuBerT       | Monophone  | Triphone(Δ+ΔΔ) | Triphone + LDA+MLLT | Triphone + LDA+MLLT+SAT |
| ---------       | ----- | ----- | ----- |----- |
| base (25fps)    | 44.59 | 22.23  | 15.39  | 13.78 |
| large (25fps)   | - |  - | -  | - |
| base (50fps)    | 29.58 | 12.44 | 9.81  | 8.79 |
| large (50fps)   | 19.34 |  8.74 | 7.32  | **6.36** |


### Tracking & Word Alignment Demo

[![Watch Demo](https://github.com/ialmajai/ssl-kaldi/blob/main/egs/grid/s5/demo/thumbnail.png)](https://github.com/ialmajai/ssl-kaldi/blob/main/egs/grid/s5/demo/tracking-demo.gif)


## Citation

If you use this recipe, please cite **ssl-kaldi**, [AV‑HuBERT](https://github.com/facebookresearch/av_hubert), and [Grid Corpus](https://spandh.dcs.shef.ac.uk/gridcorpus)
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

@article{cooke2006audio,
  title={An audio-visual corpus for speech perception and automatic speech recognition},
  author={Cooke, Martin and Barker, Jon and Cunningham, Stuart and Shao, Xu},
  journal={The Journal of the Acoustical Society of America},
  volume={120},
  number={5},
  pages={2421--2424},
  year={2006},
  publisher={AIP Publishing}
}
```
Contact
Author: Ibrahim Almajai (ialmajai@gmail.com)

