# Swahili ASR Recipe

Swahili automatic speech recognition using frozen SSL features
([AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT) and
[mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147)) with an
end-to-end LF-MMI (chain TDNN-F) system in Kaldi. The corpus is the
[OpenSLR 25](https://www.openslr.org/25/) Swahili broadcast-news dataset.

## Setup

The recipe is fully end-to-end; `run_e2e.sh` downloads the corpus, prepares the
data, extracts SSL features, and trains/decodes the chain model.

```
conda activate ssl-kaldi
cd egs/swahili/s5
./run_e2e.sh
```

The corpus (~9 GB) is fetched automatically on the first run:

```
wget https://openslr.trmal.net/resources/25/data_broadcastnews_sw.tar.bz2
```

To switch SSL models, edit `ssl_model` near the top of `run_e2e.sh`
(`ajesujoba/AfriHuBERT` or `utter-project/mHuBERT-147`); layer 9 is used for
both.

> **Note:** feature extraction requires the GPU compute mode to be *Default*,
> not *Exclusive_Process*. If needed: `sudo nvidia-smi -c 0`.

## Results (WER % on test)

Setup notes:

- End-to-end LF-MMI TDNN-F (`e2e_tdnnf_1a`), no i-vectors, no GMM stage.
- Reduced frame-subsampling-factor (default 3 → 2).
- SSL features extracted at layer 9.

| Features | Acoustic Model | WER (%) |
|----------|----------------|---------|
| **AfriHuBERT** (`ajesujoba/AfriHuBERT`) | E2E TDNN-F | **19.76** |
| mHuBERT-147 (`utter-project/mHuBERT-147`) | E2E TDNN-F | 19.82 |
| Kaldi MFCC baseline (SGMM2 + MMI) | SGMM2 | 26.62 |

Both SSL models give a large improvement over the conventional MFCC/SGMM2
baseline (a **26% relative WER reduction**, 26.62% → 19.76%), with AfriHuBERT
edging out mHuBERT-147. See [RESULTS](RESULTS) for the raw decode lines.

## Citation

If you use this recipe, please cite **ssl-kaldi** and the
[OpenSLR 25](https://www.openslr.org/25/) Swahili corpus.

```
@misc{ssl_kaldi,
  author       = {Ibrahim Almajai},
  title        = {ssl-kaldi: SSL features are all you need},
  year         = {2025},
  howpublished = {\url{https://github.com/ialmajai/ssl-kaldi}},
  note         = {Accessed: 2025-12}
}

@misc{openslr25_swahili,
  title        = {Swahili Broadcast News Speech Corpus (OpenSLR 25)},
  howpublished = {\url{https://www.openslr.org/25/}}
}
```

## Contact

Ibrahim Almajai — ialmajai@gmail.com
