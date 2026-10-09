# Mini LibriSpeech

This recipe demonstrates how to use self-supervised speech representations as acoustic features in Kaldi. Pretrained HuBERT models are evaluated with both hybrid chain and traditional GMM-HMM systems on Mini LibriSpeech.

## Quick Start

```
conda activate ssl-kaldi
cd egs/mini_librispeech/s5
./run.sh --stop-stage 12           # corpus + LM download, MMS-300m L14 features, GMM ladder
sudo nvidia-smi -c 3               # chain training runs 2-5 GPU jobs
./run.sh --stage 13                # chain TDNN, decoded with tglarge
```

Feature extraction needs the GPU in Default mode (`sudo nvidia-smi -c 0`).
Pick another encoder with `--ssl-model` and `--encoder-layer`.

## Results (WER % on dev-clean, tglarge LM)

| Features | Acoustic Model | WER (%) |
|-----------|---------------|---------|
| **MMS-300m (layer 14)** | **Chain TDNN** | **3.88** |
| WavLM Base+ (layer 9) | Chain TDNN | 3.97 |
| HuBERT Large (LL60K, layer 14) | Chain TDNN | 4.15 |
| HuBERT Base (LS960, layer 9) | Chain TDNN | 4.28 |
| MMS-300m (layer 14) | E2E TDNN-F | 4.48 |
| HuBERT Large (LL60K, layer 12) | Chain TDNN | 4.70 |
| XLS-R-300m (layer 12) | Chain TDNN | 4.86 |
| MMS-300m (layer 14) | Tri3b | 10.87 |
| Kaldi MFCC/filterbank baseline | CNN-TDNN | 6.79 |

## Key Findings

- SSL features outperform conventional MFCC/filterbank features by a wide margin.
- The best result is **3.88% WER** using `facebook/mms-300m` layer 14 with a chain TDNN, a **43% relative reduction** against the Kaldi baseline (6.79% → 3.88%).
- Encoder size does not decide the ranking: WavLM Base+ (95 M parameters) matches MMS-300m (315 M) to within 0.09, and XLS-R-300m trails both despite sharing MMS-300m's architecture. English is a thin slice of XLS-R's 128-language pretraining mixture, and the same encoder ties mHuBERT-147 on the Swahili recipe.
- Intermediate layers work best; layer 14 beat layer 12 for HuBERT Large.
- The Tri3b GMM-HMM system also benefits substantially from SSL representations, so the embeddings carry strong phonetic information on their own.
- The three encoder rows measured most recently (MMS-300m, WavLM, XLS-R) share one tree, alignments and lattices, so they differ only in features.
- See [RESULTS](RESULTS) for the raw decode lines.

## Tested SSL Models

- `facebook/mms-300m`
- `microsoft/wavlm-base-plus`
- `facebook/hubert-base-ls960`
- `facebook/hubert-large-ll60k`
- `facebook/wav2vec2-xls-r-300m`

## Alignment Accuracy

![Alignment Sample](docs/misc/spectrogram.png)

[Play audio](https://rawcdn.githack.com/ialmajai/ssl-kaldi/02d212f330085da3ac7bce5d014dd1027dc6ae5a/egs/mini_librispeech/s5/docs/index.html)
