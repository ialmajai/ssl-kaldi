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

Then run the recipe. The 5.83% system uses large AV-HuBERT, layer 13:

```
./run.sh --stage -1 --model-size large   # downloads, features, GMM-SAT, fMLLR TDNN-F chain
```

For the Lombard GRID results, fetch the front-view videos and decode them:

```
mkdir -p lombard-grid
wget -O lombard-grid/lombardgrid_front.zip "https://zenodo.org/records/3736465/files/lombardgrid_front.zip?download=1"
unzip -q lombard-grid/lombardgrid_front.zip -d lombard-grid   # -> lombard-grid/lombardgrid/front
./run.sh --stage 14 --model-size large
```

## Unseen-speaker performance

| System            | Setting         | Grammar/LM         | Data augmentation                        | WER (unseen speaker) |
| ----------------- | --------------- | ------------------ | ---------------------------------------- | -------------------- |
| LipNet            | unseen speakers | Fixed GRID grammar | Spatial/temporal jitter, mirroring, etc. | 11.4% |
| SSL-Kaldi GMM-HMM (SAT) | unseen speakers | Fixed GRID grammar | None                                | 6.03% |
| SSL-Kaldi TDNN-F chain (fMLLR) | unseen speakers | Fixed GRID grammar | None                          | **5.83%** |

On [Lombard GRID](https://spandh.dcs.shef.ac.uk/avlombard/) (out-of-domain, unseen in training), all
but one system score better than in-domain. Lombard speakers talk with more effort and clearer
articulation. Speaker adaptation decides how much of that gain a system keeps:

| System | base, layer 4 (in-domain → Lombard) | large, layer 13 (in-domain → Lombard) |
| ------ | ------------------------------------ | --------------------------------------- |
| GMM-SAT               | 7.30% → 5.39% | 6.03% → 4.73% |
| TDNN-F chain on fMLLR features | 6.72% → 5.32% | 5.83% → **4.66%** |
| Flat-start e2e chain, renorm + no-relu | 7.51% → 7.16% | 6.43% → 5.41% |
| Flat-start e2e chain, + full CMVN | 7.11% → 7.35% | 6.41% → 5.76% |

Without CMVN, the flat-start e2e chain gains 0.35 abs on base L4 and 1.02 on large L13. GMM-SAT and
the TDNN-F chain built on it apply a per-speaker fMLLR transform and gain 1.2-1.9 abs.

Full CMVN gives the e2e chain a small in-domain gain and a small Lombard loss at both encoder sizes.
Both effects sit within run-to-run noise: retraining the base-l4 CMVN chain moved its in-domain WER by
about 0.4 abs (RESULTS, 2026-10 rows).

## More Results (WER)

- 30-dimensional PCA for the HMM-GMM systems.
- Features upsampled from 25 to 50 fps.
- Mouth crops aligned to a mean face instead of the earlier centroid crop.
- Layer 4 for base AV-HuBERT and layer 13 for large, picked by a layer sweep over the old
  defaults of 9 and 12. A full mono→SAT ladder confirms both.
- Speaker adaptation (LDA+MLLT+SAT) beats TDNN systems trained on raw features, as this
  [paper](https://ueaeprints.uea.ac.uk/id/eprint/63479/1/Accepted_manuscript.pdf) found for visual
  speech features.

| AV-HuBERT           | Monophone | Triphone (Δ+ΔΔ) | Triphone + LDA+MLLT | Triphone + LDA+MLLT+SAT |
| -------------------- | --------- | --------------- | ------------------- | ----------------------- |
| base, layer 4 (50fps)  | 19.00     | 9.81            | 8.90                 | 7.30 |
| large, layer 13 (50fps) | 19.31     | 8.55            | 7.08                 | **6.03** |

We left out 25 fps rows. We measured them at the old default layers (9 base, 12 large) with the
earlier centroid crop, so they don't compare with the numbers above. Standard Kaldi HMM topologies
lose accuracy on 40 ms frames, and the table uses 50 fps throughout.

### Chain results

We trained two chain architectures on these features:

| Chain architecture | base, layer 4 | large, layer 13 |
| ------------------- | ------------- | ---------------- |
| Flat-start e2e (raw features, renorm + no-relu input layer + full CMVN) | 7.11% | 6.41% |
| TDNN-F on fMLLR features (speaker-adapted, GMM-SAT-derived, `local/chain/run_tdnn_fmllr_tdnnf.sh`) | **6.72%** | **5.83%** |

Adding full CMVN to the renorm+no-relu e2e chain moves base-layer-4 from 7.51% to 7.11% and
large-layer-13 from 6.43% to 6.41%. The base gain is about the size of run-to-run noise.

At both encoder sizes, the TDNN-F chain on fMLLR features beats the flat-start e2e chain and the
GMM-SAT system under it. Large layer 13 gives the recipe's best WER, 5.83%.

### Tracking & Word Alignment Demo

[![Watch Demo](demo/thumbnail.png)](demo/tracking-demo.gif)

### Interactive Lipreading Demo

The demo runs a fine-tuned variant of this recipe on your own clip: base AV-HuBERT with a rank-32
LoRA adapter (`input/avhubert_base_ctc_lora.pt`, layer 10) -> flat-start e2e TDNN-F chain trained on
clean plus one augmented copy (`run_demo_base_lor10_aug2x.sh`). It gets 6.33% WER on the
unseen-speaker test set and 5.42% on the Lombard eval speakers. Each word's confidence fuses the
lattice MBR posterior with a TruCLeS confidence model (`run_demo_base_lor10_conf.sh`).

```
# CLI: print the recognised text for one video
local/demo/demo.py grid-corpus/s1/bbaf2n.mpg

# Gradio web UI: upload or record a clip, see the mouth ROI + prediction
local/demo/demo.py --ui
```

The demo knows GRID's fixed 6-word grammar and nothing else (command · colour · preposition ·
letter · digit · adverb, e.g. "place green with h eight now"). Clips must run 2-15 s (up to 20 s
from a webcam) and show one frontal face.

## Citation

If you use this recipe, please cite **ssl-kaldi**, [AV-HuBERT](https://github.com/facebookresearch/av_hubert), the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus) and, for the Lombard results, the [Lombard GRID corpus](https://spandh.dcs.shef.ac.uk/avlombard/).

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

@article{alghamdi2018lombard,
  title   = {A corpus of audio-visual {Lombard} speech with frontal and profile views},
  author  = {Alghamdi, Najwa and Maddock, Steve and Marxer, Ricard and Barker, Jon and Brown, Guy J.},
  journal = {The Journal of the Acoustical Society of America},
  volume  = {143},
  number  = {6},
  pages   = {EL523--EL529},
  year    = {2018}
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
