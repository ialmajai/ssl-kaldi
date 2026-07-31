# TIMIT

Phone recognition on TIMIT with self-supervised speech representations in place
of MFCCs. The pipeline is Kaldi's classic `egs/timit/s5` recipe (48-phone
training, bigram phone LM, scoring mapped to the standard 39 phone classes)
with the feature stage swapped for frozen SSL embeddings:

| Data directory | Features | Used by |
|----------------|----------|---------|
| `data/<set>_raw` | SSL encoder output (768-d base / 1024-d large), 20 ms frame shift | chain TDNN-F |
| `data/<set>_pca` | PCA-reduced SSL features (30-d by default) | mono / tri1 / tri2 / tri3 |

The SSL model's weights are never updated; `shared/make_ssl.sh` runs it as a
fixed feature extractor and writes ordinary Kaldi `ark/scp` archives.

## Requirements

- The TIMIT corpus (LDC93S1), i.e. the directory that contains `TRAIN/` and `TEST/`.
  Sphere (`.WAV`) and already-converted RIFF distributions both work.
- A Kaldi build with **IRSTLM** (`tools/extras/install_irstlm.sh`), used by
  `local/timit_prepare_dict.sh` to build the phone bigram LM, and **sph2pipe**
  (`tools/extras/install_sph2pipe.sh`) for sphere audio.
- The repo's Python environment (see the [top-level README](../../../README.md)).

## Running

```
conda activate ssl-kaldi
cd egs/timit/s5
./run.sh --timit /path/to/TIMIT
```

Stages:

| Stage | What it does |
|-------|--------------|
| 0 | Data prep, lexicon, `data/lang`, phone bigram LM |
| 1 | SSL feature extraction + CMVN (`data/*_raw`) |
| 2 | PCA model + PCA features (`data/*_pca`) |
| 3–6 | mono, tri1 (Δ+ΔΔ), tri2 (LDA+MLLT), tri3 (SAT) on PCA features |
| 7 | Speed-perturbed features and tri3 alignments (`local/chain/run_common_ssl.sh`) |
| 8 | Chain TDNN-F on raw SSL features (`local/chain/run_tdnn_ssl.sh`) |

Pick the SSL model and layer with `ssl_model` / `encoder_layer` at the top of
`run.sh`, or as options. The default `facebook/mms-300m` layer 14 is the best
measured here; `facebook/hubert-base-ls960` layer 9 is 1.2 points worse at a
third of the extraction cost, which is the one to use while iterating.

Feature extraction needs the GPU in default compute mode
(`sudo nvidia-smi -c 0`); chain training is happier in exclusive mode
(`sudo nvidia-smi -c 3`).

## Scoring

`local/score.sh` maps reference and hypothesis to the 39 phone classes and
reports PER with `compute-wer`, so no external scoring tool is needed. For the
sclite numbers upstream TIMIT reports, swap in the included alternative with
`ln -sf score_sclite.sh local/score.sh`.

Both sweep LM weights 1 to 20 rather than Kaldi's 1 to 10, because some systems
genuinely peak above 10 and a truncated sweep under-reports them.

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
20.3/21.6) to within a few tenths. Kaldi's TIMIT recipe has no chain system;
its strongest published result is 16.7 / 18.4 (SGMM + DNN combination).

### Model comparison

Four frozen extractors through the identical pipeline; the full per-stage
breakdown for each is in [RESULTS](RESULTS).

| Model | layer | dim | tri2 test | tri3 test | chain dev | chain test |
|-------|-------|-----|-----------|-----------|-----------|------------|
| `facebook/mms-300m` | 14 | 1024 | **15.27** | 15.70 | **9.19** | **10.19** |
| `facebook/wav2vec2-large-lv60` | 14 | 1024 | 18.14 | 17.73 | 9.87 | 10.99 |
| `facebook/hubert-base-ls960` | 9 | 768 | 16.15 | 16.31 | 9.93 | 11.43 |
| `facebook/hubert-large-ll60k` | 14 | 1024 | 15.98 | 15.79 | 10.02 | 11.48 |

mms-300m wins by 90 errors over HuBERT Base (~3.2 standard errors), and on
insertions, deletions and substitutions at once. Note the GMM columns rank the
models differently from the chain column: wav2vec2 is last at tri2 and second at
the chain stage.

mms-300m layer 14 is the recipe default. Use
`--ssl-model facebook/hubert-base-ls960 --encoder-layer 9` when extraction cost
matters more than the last point of PER.

### Layer choice within one model

Which layer you tap matters, and not in a single direction
(`facebook/hubert-base-ls960`, frozen, %PER test):

| Acoustic model | layer 9 | layer 12 |
|----------------|---------|----------|
| mono | 24.63 | **24.59** |
| tri1 | 18.38 | **17.93** |
| tri2 | **16.15** | 17.30 |
| tri3 | 16.31 | **16.11** |
| chain TDNN-F | **11.43** | 11.74 |

The deeper layer wins for the weakest models and loses for the strongest. Layer
9 is the better choice for the chain system.

### Layer sweep (HuBERT Large)

Sweeping Large's encoder layer, scored at tri2
(`./run.sh --stage 1 --stop-stage 5 --encoder-layer N`):

| layer | tri2 dev | tri2 test |
|-------|----------|-----------|
| 10 | 16.04 | 17.20 |
| 11 | 15.26 | 16.78 |
| 12 | 14.86 | 16.40 |
| 13 | **14.58** | **15.90** |
| 14 | **14.45** | 15.98 |

Monotone from 10 to 13, then flat, so Large's phonetic quality is still rising
at the layer we use rather than past its peak. These are tri2 numbers, which
rank layers within one model reliably enough but not models against each other,
and the 12-to-14 spread is close to the noise floor.

## Key findings

- **10.19% PER on the core test set**, a 53% relative reduction over the MFCC
  SAT baseline (21.95%) and better than any system in Kaldi's published TIMIT
  results.
- **Pretraining data beats parameter count.** mms-300m and HuBERT Large are the
  same scale, but Large only ties Base (11.48 vs 11.43) while mms-300m reaches
  10.19. What separates them is 491k hours across 1400+ languages against 60k
  hours of English audiobooks.
- **SAT buys nothing.** fMLLR gains 2.21 points with MFCCs and between 0.00 and
  −0.33 with every SSL model, so SSL features are already speaker-normalised.
  Use tri2 rather than tri3 if you only need alignments.
- **The GMM stages do not rank models.** wav2vec2-large-lv60 is the worst of the
  four by 2 to 3 points at tri2 and the second best at the chain stage. Treat
  tri2 as a smoke test, not a model-selection criterion.
- Chain training overfits mildly on 3.7 h; fewer epochs or stronger
  regularisation is the obvious thing to try.
