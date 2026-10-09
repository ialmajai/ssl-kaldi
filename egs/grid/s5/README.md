# Lipreading Recipe using AV-HuBERT and the GRID Corpus

Visual speech recognition (lipreading) on the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus)
using frozen [AV-HuBERT](https://github.com/facebookresearch/av_hubert) visual features in Kaldi
GMM-HMM pipelines.

## AV-HuBERT Setup

```
conda activate ssl-kaldi
pip install pip==24.0
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

| System            | Setting         | Grammar/LM         | Data augmentation                        | WER (unseen speaker) |
| ----------------- | --------------- | ------------------ | ---------------------------------------- | -------------------- |
| LipNet            | unseen speakers | Fixed GRID grammar | Spatial/temporal jitter, mirroring, etc. | 11.4% |
| SSL-Kaldi GMM-HMM (SAT) | unseen speakers | Fixed GRID grammar | None                                | 6.03% |
| SSL-Kaldi TDNN-F chain (fMLLR) | unseen speakers | Fixed GRID grammar | None                          | **5.83%** |

On [Lombard GRID](https://spandh.dcs.shef.ac.uk/gridcorpus) (out-of-domain, unseen during training), every
system scores better than its own in-domain result, consistent with the Lombard effect producing more
effortful, clearer speech -- but the size of that gap depends heavily on speaker adaptation:

| System | base, layer 4 (in-domain → Lombard) | large, layer 13 (in-domain → Lombard) |
| ------ | ------------------------------------ | --------------------------------------- |
| GMM-SAT               | 7.30% → 5.39% | 6.03% → **4.73%** |
| Flat-start e2e chain, renorm + no-relu | 7.51% → 7.16% | 6.43% → 5.41% |
| Flat-start e2e chain, + full CMVN | 7.11% → 7.35% | 6.41% → 5.76% |

(TDNN-F chain Lombard numbers not yet measured.)

The flat-start e2e chain barely benefits from Lombard's clearer speech (base-l4: only 0.35% abs without
CMVN), while the GMM-SAT system, which estimates a per-speaker fMLLR transform, gains roughly 1.3% abs.

**Full CMVN trades in-domain accuracy for OOD robustness, consistently, at both encoder sizes**: it
improves the in-domain e2e result but makes the *same* model worse on Lombard, reversing which e2e
variant is best depending on whether the target domain matches training. Not just noise -- the sign
flips the same way for both base-l4 and large-L13. Since full CMVN normalizes mean and variance
per-utterance rather than per-corpus, this isn't ordinary train/test feature mismatch; more likely the
network leans on some GRID-specific artifact of the normalized per-utterance distribution that doesn't
hold under Lombard's different loudness/effort profile.

## More Results (WER)

- 30-dimensional PCA for the HMM-GMM systems.
- Features upsampled to 50 fps give better results.
- Mean-face-aligned features (see below), not the earlier centroid crop.
- A layer sweep (below) shows layer 4 is best for base AV-HuBERT (not layer 9) and layer 13
  is best for large AV-HuBERT (not layer 12) -- both re-validated with a full mono→SAT ladder.
- Speaker adaptation (MLLT+SAT) outperforms TDNN systems trained on raw features. The power of such
  adaptation for visual speech features has already been demonstrated in this
  [paper](https://ueaeprints.uea.ac.uk/id/eprint/63479/1/Accepted_manuscript.pdf).

| AV-HuBERT           | Monophone | Triphone (Δ+ΔΔ) | Triphone + LDA+MLLT | Triphone + LDA+MLLT+SAT |
| -------------------- | --------- | --------------- | ------------------- | ----------------------- |
| base, layer 4 (50fps)  | 19.00     | 9.81            | 8.90                 | 7.30 |
| large, layer 13 (50fps) | 19.31     | 8.55            | 7.08                 | **6.03** |

25fps rows are omitted here: they were only ever measured at the old default layers
(9 base / 12 large) under the earlier centroid crop and are not comparable to the numbers
above. Standard Kaldi HMM topologies handle 40ms frames poorly, which is why every row
here uses 50fps.

### Chain results

Two chain architectures were tried on top of the layer-sweep features:

| Chain architecture | base, layer 4 | large, layer 13 |
| ------------------- | ------------- | ---------------- |
| Flat-start e2e (raw features, renorm + no-relu input layer + full CMVN) | 7.11% | 6.41% |
| TDNN-F on fMLLR features (speaker-adapted, GMM-SAT-derived, `run_tdnn_fmllr_tdnnf.sh`) | **6.72%** | **5.83%** |

Adding full CMVN to the renorm+no-relu e2e chain helps base-layer-4 substantially (7.51% → 7.11%) but
barely moves large-layer-13 (6.43% → 6.41%) -- the larger encoder's deeper features may already be
better-normalized by the renorm-component alone, leaving less for CMVN to correct.

The fMLLR-feature TDNN-F chain beats both the flat-start e2e chain and the GMM-SAT system it is
built on, for both encoder sizes -- the large-layer-13 result (5.83%) is the best result in this
recipe.

### Base AV-HuBERT layer sweep

WER against encoder layer for base AV-HuBERT (50fps, mean-face-aligned features), at every
stage of the GMM ladder. Layer 4 is a clear local minimum, beating the previously-used
layer 9 by 0.73 abs (~9% relative) at the LDA+MLLT+SAT stage.

![Layer sweep: WER vs AV-HuBERT base encoder layer](fig_layer_sweep.png)

### Large AV-HuBERT layer sweep

Same sweep for large AV-HuBERT (50fps, mean-face-aligned features), layers 8-14. The
minimum is shallow and noisy across 9-13 (spread of only 0.18 abs) with clear degradation
at both edges; layer 13 is best, beating the previously-used layer 12 by 0.11 abs.

![Layer sweep: WER vs AV-HuBERT large encoder layer](fig_layer_sweep_large.png)

### Tracking & Word Alignment Demo

[![Watch Demo](demo/thumbnail.png)](demo/tracking-demo.gif)

### Interactive Lipreading Demo

Runs this recipe's best base-model system end to end on your own clip: base AV-HuBERT (layer 4)
-> PCA-30 -> Kaldi GMM-SAT (per-utterance fMLLR transform) -> fMLLR TDNN-F chain.

```
# CLI -- print the recognised text for one video
local/demo/demo.py grid-corpus/s1/bbaf2n.mpg

# Gradio web UI -- upload or record a clip, see the mouth ROI + prediction
local/demo/demo.py --ui
```

Only understands GRID's fixed 6-word grammar (command · colour · preposition · letter · digit ·
adverb, e.g. "place green with h eight now"); clips must be 2-15s (up to 20s from a webcam) with a
single frontal face.

## Citation

If you use this recipe, please cite **ssl-kaldi**, [AV-HuBERT](https://github.com/facebookresearch/av_hubert), and the [GRID corpus](https://spandh.dcs.shef.ac.uk/gridcorpus).

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
