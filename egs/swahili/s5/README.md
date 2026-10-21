# Swahili ASR Recipe

Swahili automatic speech recognition using frozen SSL features
([AfriHuBERT](https://huggingface.co/ajesujoba/AfriHuBERT) and
[mHuBERT-147](https://huggingface.co/utter-project/mHuBERT-147)) with an
end-to-end LF-MMI (chain TDNN-F) system in Kaldi. The corpus is the
[OpenSLR 25](https://www.openslr.org/25/) Swahili broadcast-news dataset.

## Quick Start

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
both. `facebook/wav2vec2-xls-r-300m` also works, tapped at layer 12.

> **Note:** feature extraction requires the GPU compute mode to be *Default*,
> not *Exclusive_Process*. If needed: `sudo nvidia-smi -c 0`.

## Results (WER % on test)

Setup notes:

- End-to-end LF-MMI TDNN-F, no i-vectors, no GMM stage.
- Reduced frame-subsampling-factor (default 3 → 2).
- SSL features extracted at layer 9, except XLS-R at layer 12.
- mHuBERT-147 and XLS-R use the current script defaults (renorm-linear input,
  10 TDNN-F layers); AfriHuBERT uses the older 12-layer relu-batchnorm
  network. All rows share one tree and phone LM.

| Features | Network | WER (%) |
|----------|---------|---------|
| AfriHuBERT (`ajesujoba/AfriHuBERT`) | 12L relu-batchnorm | 19.76 |
| Kaldi MFCC baseline (SGMM2 + MMI) | SGMM2 | 26.62 |
| **mHuBERT-147** | **10L renorm-linear** | **19.47** |
| XLS-R-300m (`facebook/wav2vec2-xls-r-300m`, L12) | 10L renorm-linear | 19.59 |

Every SSL model beats the conventional MFCC/SGMM2 baseline by a wide margin
(26.62% → 19.47%, a 27% relative reduction). The three multilingual encoders
land within 0.29 of each other, which is inside this test set's confidence
interval of roughly ±0.8, so treat their ordering as unresolved rather than
as a ranking. See [RESULTS](RESULTS) for the raw decode lines.

### Fine-tuned encoder

`run_ctc_finetune.sh` fine-tunes mHuBERT-147 with a character CTC head, then
reads layer 9 of the fine-tuned encoder into the same chain. Run it after
`run_e2e.sh`. Two held-out training sessions choose the checkpoint and the
CTC decoder's LM weights, so the test set chooses nothing.

```
./run_ctc_finetune.sh              # LoRA r32 on attention + FFN
./run_ctc_finetune.sh --arm full   # every encoder weight trainable
```

| Fine-tuning | CTC greedy | CTC + trigram | Chain |
|-------------|------------|---------------|-------|
| none (frozen) | - | - | 19.47 |
| LoRA r32 | 29.16 | 19.18 | **19.07** |
| full | 24.63 | 19.72 | 19.36 |

## Citation

If you use this recipe, please cite **ssl-kaldi** and the
[OpenSLR 25](https://www.openslr.org/25/) Swahili corpus.

```
@misc{ssl_kaldi,
  author       = {Ibrahim Almajai},
  title        = {ssl-kaldi: self-supervised speech features for Kaldi ASR recipes},
  year         = {2025},
  howpublished = {\url{https://github.com/ialmajai/ssl-kaldi}},
  note         = {Accessed: 2026-08}
}

@misc{openslr25_swahili,
  title        = {Swahili Broadcast News Speech Corpus (OpenSLR 25)},
  howpublished = {\url{https://www.openslr.org/25/}}
}
```

## Contact

Ibrahim Almajai (ialmajai@gmail.com)
