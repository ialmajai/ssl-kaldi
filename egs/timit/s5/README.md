# TIMIT

Phone recognition on TIMIT with frozen self-supervised speech representations
in place of MFCCs. The recipe follows Kaldi's `egs/timit/s5`: 48-phone training,
a bigram phone LM, and scoring mapped to the standard 39 phone classes.

## Quick Start

```
conda activate ssl-kaldi
cd egs/timit/s5
./run.sh --timit /path/to/TIMIT --stop-stage 7   # features, GMM ladder, chain prep
sudo nvidia-smi -c 3                             # chain training wants exclusive mode
./run.sh --timit /path/to/TIMIT --stage 8        # chain TDNN-F
```

The recipe needs IRSTLM and sph2pipe in the Kaldi build (see [Requirements](#requirements)).

## Feature layout

| Data directory | Features | Used by |
|----------------|----------|---------|
| `data/<set>_raw` | SSL encoder output (768-d base / 1024-d large), 20 ms frame shift | chain TDNN-F |
| `data/<set>_pca` | PCA-reduced SSL features (30-d by default) | mono / tri1 / tri2 / tri3 |

`shared/make_ssl.sh` runs the SSL model as a frozen feature extractor and writes
ordinary Kaldi `ark/scp` archives.

## Requirements

- The TIMIT corpus (LDC93S1), i.e. the directory that contains `TRAIN/` and `TEST/`.
  Sphere (`.WAV`) and already-converted RIFF distributions both work.
- A Kaldi build with **IRSTLM** (`tools/extras/install_irstlm.sh`), used by
  `local/timit_prepare_dict.sh` to build the phone bigram LM, and **sph2pipe**
  (`tools/extras/install_sph2pipe.sh`) for sphere audio.
- The repo's Python environment (see the [top-level README](../../../README.md)).

## Stages

| Stage | What it does |
|-------|--------------|
| 0 | Data prep, lexicon, `data/lang`, phone bigram LM |
| 1 | SSL feature extraction + CMVN (`data/*_raw`) |
| 2 | PCA model + PCA features (`data/*_pca`) |
| 3–6 | mono, tri1 (Δ+ΔΔ), tri2 (LDA+MLLT), tri3 (SAT) on PCA features |
| 7 | Speed-perturbed features and tri3 alignments (`local/chain/run_common_ssl.sh`) |
| 8 | Chain TDNN-F on raw SSL features (`local/chain/run_tdnn_ssl.sh`) |

Pick the SSL model and layer with `ssl_model` / `encoder_layer` at the top of
`run.sh`, or as options. The default, `facebook/mms-300m` layer 14, gives the
best PER we measured. `facebook/hubert-base-ls960` layer 9 costs a third as much
to extract and loses 1.2 points on test; use it while you iterate.

Feature extraction needs the GPU in default compute mode
(`sudo nvidia-smi -c 0`). Switch to exclusive mode (`sudo nvidia-smi -c 3`) for
chain training.

## Scoring

`local/score.sh` maps reference and hypothesis to the 39 phone classes and
reports PER with `compute-wer`, so you need no external scoring tool. For the
sclite numbers upstream TIMIT reports, swap in the included alternative with
`ln -sf score_sclite.sh local/score.sh`.

Both scripts sweep LM weights 1 to 20 instead of Kaldi's 1 to 10. Some systems
here peak above 10, and a 1-to-10 sweep would under-report them.

Collect results with `bash RESULTS dev` / `bash RESULTS test`.

## Results (%PER, 39 phone classes, core test set)

`facebook/mms-300m` layer 14, the recipe default, with PCA-30 for the GMM
stages:

| Acoustic model | SSL dev | SSL test | MFCC dev | MFCC test |
|----------------|---------|----------|----------|-----------|
| mono | 23.43 | 24.75 | 31.78 | 32.13 |
| tri1 (Δ+ΔΔ) | 17.13 | 18.41 | 24.94 | 26.36 |
| tri2 (LDA+MLLT) | 14.43 | 15.27 | 22.85 | 23.34 |
| tri3 (SAT) | 14.68 | 15.70 | 20.31 | 21.95 |
| **chain TDNN-F** | **9.19** | **10.19** | n/a | n/a |

The MFCC columns are `run-mfcc.sh` run locally, so both halves of the table
share a data prep, speaker lists and scoring script. They reproduce the numbers
published in Kaldi's `egs/timit/s5/RESULTS` (31.7/32.7, 25.1/25.6, 23.0/23.7,
20.3/21.6) to within 0.8. Kaldi's TIMIT recipe has no chain system;
its strongest published result is 16.7 / 18.4 (SGMM + DNN combination).

### Model comparison

We ran four frozen encoders through the same pipeline. [RESULTS](RESULTS) has
the per-stage breakdown for each.

| Model | layer | dim | tri2 test | tri3 test | chain dev | chain test |
|-------|-------|-----|-----------|-----------|-----------|------------|
| `facebook/mms-300m` | 14 | 1024 | **15.27** | 15.70 | **9.19** | **10.19** |
| `facebook/wav2vec2-large-lv60` | 14 | 1024 | 18.14 | 17.73 | 9.87 | 10.99 |
| `facebook/hubert-base-ls960` | 9 | 768 | 16.15 | 16.31 | 9.93 | 11.43 |
| `facebook/hubert-large-ll60k` | 14 | 1024 | 15.98 | 15.79 | 10.02 | 11.48 |

On chain test, mms-300m makes 90 fewer errors than HuBERT Base (about 3.2
standard errors) and fewer insertions, deletions and substitutions each. The GMM
columns rank the models differently from the chain columns: wav2vec2 comes last
at tri2 and second at the chain stage.
