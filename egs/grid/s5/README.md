# Lipreading Recipe using AV-HuBERT and the GRID Corpus

Visual speech recognition (lipreading) on the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus)
using frozen [AV-HuBERT](https://github.com/facebookresearch/av_hubert) visual features in Kaldi
GMM-HMM and chain pipelines.

## Quick Start

Install AV-HuBERT and dlib into the repo environment. We have run AV-HuBERT's fairseq on
Python 3.8 and 3.10 (the Docker image) but not on 3.11, so create the environment with
`python=3.10`.

```
conda activate ssl-kaldi
cd egs/grid/s5
pip install pip==24.0
pip install -r requirements.txt
git clone https://github.com/facebookresearch/av_hubert.git --depth=1
cd av_hubert
git submodule init
git submodule update
cd fairseq
pip install --editable ./
conda install -c conda-forge dlib==19.18.0
cd ../..
```

Then run the recipe. The best system uses large AV-HuBERT, layer 11; one run gives 5.76-6.00% (mean 5.89% over three chain seeds):

```
./run.sh --stage -1 --model-size large   # downloads, features, GMM-SAT, fMLLR TDNN-F chain
```

## Unseen-speaker performance

| System            | Setting         | Grammar/LM         | Data augmentation                        | WER (unseen speaker) |
| ----------------- | --------------- | ------------------ | ---------------------------------------- | -------------------- |
| LipNet            | unseen speakers | Fixed GRID grammar | Spatial/temporal jitter, mirroring, etc. | 11.4% |
| SSL-Kaldi GMM-HMM (SAT) | unseen speakers | Fixed GRID grammar | None                                | 6.25% |
| SSL-Kaldi TDNN-F chain (fMLLR) | unseen speakers | Fixed GRID grammar | None                          | **5.89%** |

GRID test WERs in this README are on all 4,000 clips of the unseen speakers (s1, s2, s20, s22).

## More Results (WER)

- 30-dimensional PCA for the HMM-GMM systems.
- Features upsampled from 25 to 50 fps.
- Mouth crops aligned to a mean face instead of the earlier centroid crop.
- Layer 4 for base AV-HuBERT and layer 11 for large, chosen by a tri3b layer sweep scored on
  held-out dev speakers rather than on GRID test. Large layer 13, the earlier tap, came from a
  sweep scored on GRID test, so its rows below are optimistic.
- Speaker adaptation (LDA+MLLT+SAT) beats TDNN systems trained on raw features, as this
  [paper](https://ueaeprints.uea.ac.uk/id/eprint/63479/1/Accepted_manuscript.pdf) found for visual
  speech features.

| AV-HuBERT           | Monophone | Triphone (Δ+ΔΔ) | Triphone + LDA+MLLT | Triphone + LDA+MLLT+SAT |
| -------------------- | --------- | --------------- | ------------------- | ----------------------- |
| base, layer 4 (50fps)  | 19.00     | 9.85            | 8.94                 | 7.38 |
| large, layer 11 (50fps) | 19.01     | 8.97            | 7.41                 | 6.25 |
| large, layer 13 (50fps, chosen on test) | 19.35 | 8.57   | 7.13                 | 6.05 |

We left out 25 fps rows. We measured them at the old default layers (9 base, 12 large) with the
earlier centroid crop, so they don't compare with the numbers above. Standard Kaldi HMM topologies
lose accuracy on 40 ms frames, and the table uses 50 fps throughout.

### Chain results

We trained two chain architectures on these features:

| Chain architecture | base, layer 4 | large, layer 11 | large, layer 13 (chosen on test) |
| ------------------- | ------------- | --------------- | -------------------------------- |
| Flat-start e2e (raw features, renorm + no-relu input layer + full CMVN) | 7.15% | not run | 6.45% |
| TDNN-F on fMLLR features (speaker-adapted, GMM-SAT-derived, `local/chain/run_tdnn_fmllr_tdnnf.sh`) | **6.76%** | **5.89%** (3 seeds, 5.76-6.00) | 5.85% (1 run) |

Adding full CMVN to the renorm+no-relu e2e chain moves base-layer-4 from 7.55% to 7.15% and
large-layer-13 from 6.47% to 6.45%. The base gain is about the size of run-to-run noise.

At both encoder sizes, the TDNN-F chain on fMLLR features beats the flat-start e2e chain and the
GMM-SAT system under it. Large layer 11 gives the recipe's best WER, 5.89% over three chain seeds.

### Fine-tuned encoders

`run_ctc_finetune.sh` fine-tunes AV-HuBERT with a letter CTC head, either with
rank-32 LoRA or with every weight trainable, then extracts features from the
fine-tuned encoder for the flat-start e2e chain. Run it after `run.sh` stages
0-2. On one GPU, a fine-tune took 47 min (base LoRA) to 2 h 10 min (large
LoRA) in our runs, before feature extraction and chain training.

```
./run_ctc_finetune.sh                                    # LoRA, base, layer 10
./run_ctc_finetune.sh --config base_ctc_full             # every weight trainable
./run_ctc_finetune.sh --config large_ctc_lora --layer 20
```

| Encoder | Fine-tuning, layer | Trainable | GRID test |
| ------- | ------------------ | --------- | --------- |
| base  | none, L4           | 0     | 7.15 |
| base  | full, L10          | 103M  | **5.70** |
| base  | LoRA r32, L10      | 5.3M  | 6.34 |
| large | none, L11 (fMLLR chain, 3 seeds) | 0 | 5.89 |
| large | full, L20          | 325M  | 6.25 |
| large | LoRA r32, L20      | 14M   | **5.55** |

Fine-tuned rows are one fine-tune each; held-out dev speakers chose their layers.

### Mouth tracking and word alignment

[![Mouth tracking and word alignment](figures/tracking-thumbnail.png)](figures/tracking.gif)

## Citation

If you use this recipe, please cite **ssl-kaldi**, [AV-HuBERT](https://github.com/facebookresearch/av_hubert), the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus).

```
@misc{ssl_kaldi,
  author       = {Ibrahim Almajai},
  title        = {ssl-kaldi: self-supervised speech features for Kaldi ASR recipes},
  year         = {2025},
  howpublished = {\url{https://github.com/ialmajai/ssl-kaldi}},
  note         = {Accessed: 2026-08}
}

@article{shi2022avhubert,
  author  = {Bowen Shi and Wei-Ning Hsu and Kushal Lakhotia and Abdelrahman Mohamed},
  title   = {Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction},
  journal = {arXiv preprint arXiv:2201.02184},
  year    = {2022}
}

@article{cooke2006audio,
  title     = {An audio-visual corpus for speech perception and automatic speech recognition},
  author    = {Cooke, Martin and Barker, Jon and Cunningham, Stuart and Shao, Xu},
  journal   = {The Journal of the Acoustical Society of America},
  volume    = {120},
  number    = {5},
  pages     = {2421--2424},
  year      = {2006},
  publisher = {AIP Publishing}
}
```

## Contact

Ibrahim Almajai (ialmajai@gmail.com)
