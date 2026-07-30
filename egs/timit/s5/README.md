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

Pick the SSL model and layer by editing `ssl_model` / `encoder_layer` at the top
of `run.sh` (or passing them as options). The default is `facebook/mms-300m`
layer 14, the best configuration measured here.
`facebook/hubert-base-ls960` layer 9 is commented out alongside it: 1.2 points
worse but a third of the extraction cost, which is the one to use if you are
iterating on the rest of the pipeline. See
[Base vs. Large vs. mms-300m](#model-comparison) for the full comparison.

Feature extraction needs the GPU in default compute mode
(`sudo nvidia-smi -c 0`); chain training is happier in exclusive mode
(`sudo nvidia-smi -c 3`).

## Scoring

`local/score.sh` maps both reference and hypothesis to the 39 phone classes and
reports PER with `compute-wer`, so no external scoring tool is needed
(this is Kaldi's `score_basic.sh`; upstream TIMIT defaults to sclite instead).
`local/score_sclite.sh` is included for the sclite/`hubscr.pl` numbers if you
have `tools/sctk` built; the decode scripts always call `local/score.sh`, so
swap it in with `ln -sf score_sclite.sh local/score.sh`.

Both scripts sweep LM weights 1 to 20, wider than Kaldi's default of 10, because
some systems genuinely peak above 10: a more confident acoustic model needs a
heavier LM weight to balance it, and truncating the sweep at 10 silently
under-reports them. All numbers in this README were produced with the wider
range.

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

**mms-300m wins**, and it is the one comparison here that is statistically
solid: 735 test errors against HuBERT Base's 825, roughly 3.2 standard errors.
It also wins on insertions, deletions and substitutions at once (174/157/404
against 209/164/452), so this is a real acoustic gain and not a better
insertion/deletion operating point.

**Model size is not the variable.** mms-300m (~300 M parameters) and HuBERT
Large (~317 M) are the same scale, yet Large only ties Base (11.48 against
11.43, a 3-error difference). What separates them is the pretraining corpus:
~491 k hours across 1400+ languages against 60 k hours of English audiobooks.

**The GMM stages do not rank the models.** wav2vec2-large-lv60 is the worst of
the four by 2 to 3 points at every GMM stage and the second best at the chain
stage. Its information is evidently not recoverable from 30 PCA components with
diagonal-covariance GMMs, but a network with all 1024 dimensions finds it. Treat
tri2 as a smoke test, not as a model-selection criterion.

mms-300m layer 14 is the recipe default. Use
`--ssl-model facebook/hubert-base-ls960 --encoder-layer 9` when extraction cost
matters more than the last point of PER.

### Layer choice within one model

Which layer you tap matters, and not in a single direction. Comparing
`facebook/hubert-base-ls960` at layer 9 against layer 12, both frozen (%PER
test):

| Acoustic model | layer 9 | layer 12 |
|----------------|---------|----------|
| mono | 24.63 | **24.59** |
| tri1 | 18.38 | **17.93** |
| tri2 | **16.15** | 17.30 |
| tri3 | 16.31 | **16.11** |
| chain TDNN-F | **11.43** | 11.74 |

The deeper layer wins for the weakest models and loses once the acoustic model
is strong enough to exploit the richer mid-stack representation. Layer 9 is the
better choice for the chain system, which is the one that matters.

### Layer sweep (HuBERT Large)

Layer 14 is not the reason. Sweeping Large's encoder layer, scored at tri2
(`./run.sh --stage 1 --stop-stage 5 --encoder-layer N`):

| layer | tri2 dev | tri2 test |
|-------|----------|-----------|
| 10 | 16.04 | 17.20 |
| 11 | 15.26 | 16.78 |
| 12 | 14.86 | 16.40 |
| 13 | **14.58** | **15.90** |
| 14 | **14.45** | 15.98 |

A monotone improvement from 10 to 13, then flat: layers 13 and 14 are within
noise of each other and swap ranks between dev and test. Large's phonetic
quality is still rising at the layer we use, not past its peak, so a shallower
layer will not recover a Large advantage. The same ordering holds at mono and
tri1, so it is not a decoding artefact. Two caveats, and the first turned out to
matter: these are PCA-30 tri2 numbers, and the model comparison above shows tri2
rank can be wrong by 2 to 3 points across models, so read this sweep as
depth-within-one-model only. The whole spread
from 12 to 14 is about 0.5 points, close to the noise floor on 7215 phones.

## Key findings

- **10.19% PER on the core test set** with a frozen mms-300m (11.43% with
  HuBERT Base): a 53% relative reduction over the MFCC SAT baseline (21.6%),
  and 45% better than the best system in Kaldi's published TIMIT results.
- Every GMM stage improves substantially, and the SSL *monophone* system
  (24.63%) is already close to the MFCC *triphone* baseline (25.6%). The
  phonetic information is in the features, not the acoustic model.
- **SAT buys nothing here, and that is a property of the features, not of
  TIMIT.** Comparing tri3's speaker-independent pass against its fMLLR-adapted
  pass isolates the adaptation exactly: same model, same tree, same graph.

  | features | tri3 `.si` test | tri3 adapted test | fMLLR gain |
  |----------|-----------------|-------------------|------------|
  | MFCC | 24.16 | 21.95 | **+2.21** |
  | HuBERT Base L9 | 15.98 | 16.31 | −0.33 |
  | HuBERT Large L14 | 15.52 | 15.79 | −0.27 |
  | mms-300m L14 | 15.70 | 15.70 | 0.00 |

  MFCCs gain 2.2 points from fMLLR on this corpus, with roughly 8 utterances
  (~25 s) per speaker, so the corpus supports speaker adaptation fine. All three
  SSL models gain nothing or slightly less than nothing: the representations are
  already largely speaker-normalised, leaving fMLLR only estimation variance to
  contribute. Use tri2 rather than tri3 if you only need alignments.
- **Pretraining data beats parameter count.** HuBERT Large (~317 M) only ties
  Base at the chain stage (11.48 vs 11.43 test) for triple the extraction cost,
  while mms-300m at the same scale as Large reaches 10.19. The models that
  differ in size tie; the model trained on far more and far more diverse audio
  wins.
- Chain training overfits mildly on this amount of data (final train objective
  −0.051 vs −0.258 on the validation set; −0.038 vs −0.220 for Large, i.e. the
  larger model overfits slightly more). Fewer epochs or stronger regularisation
  is the obvious thing to try.

## Notes

- SSL features come out at a 20 ms frame shift, so the chain script uses
  `frame_subsampling_factor=2` (40 ms output frames), matching the other
  recipes. Dropping to `1` (20 ms output) was tried on the Large L14 features
  and is *worse*: 10.79 / 11.70 against 10.02 / 11.48. Insertions fall (174 to
  111 at the best LM weight) but deletions and substitutions both rise, so the
  coarser rate appears to act as useful smoothing rather than as a bottleneck.
  Note that Kaldi scales `num_archives_to_process` by the subsampling factor,
  so at `1` the same `--num-epochs` is half the parameter updates; that run was
  therefore also undertrained relative to the default.
- The GMM stages use PCA-30 features because full 768/1024-d embeddings are too
  high-dimensional for diagonal-covariance GMMs; the chain model consumes the
  full-dimensional features directly.
- SA utterances are excluded (as upstream), silence has `--sil-prob 0.0`, and
  `sil` is a scored word. Do not "fix" these, they are what makes the numbers
  comparable to the published TIMIT literature.
