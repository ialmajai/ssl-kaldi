# ssl-kaldi-feats

Kaldi `ark`/`scp` features from any HuggingFace speech encoder. **No Kaldi
installation required** - archives are written in pure Python, so the output is
consumable by Kaldi, ESPnet, or anything else that reads the format, on a
machine that has never compiled Kaldi.

```bash
pip install ssl-kaldi-feats
```

## Use

```bash
# single layer, from a Kaldi wav.scp (the "cmd |" form works too)
ssl-kaldi-feats --model facebook/hubert-base-ls960 --layer 9 \
    --wav-scp data/train/wav.scp --output-dir feats/train

# or from a plain list of audio files
ssl-kaldi-feats --model microsoft/wavlm-large --layer 16 \
    --file-list files.txt --output-dir feats/train
```

Output is a directory Kaldi will accept as-is:

```
feats/train/feats.scp  feats.ark  utt2num_frames  utt2dur  frame_shift
```

As a library:

```python
from ssl_kaldi_feats import SSLExtractor

ex = SSLExtractor("facebook/mms-300m", layers=14)
feats = ex.extract_one(waveform)          # (frames, 1024) float32, 20 ms shift
```

## Sweeping layers costs one forward pass, not one per layer

```bash
ssl-kaldi-feats --model facebook/mms-300m --layers 10,12,14,16,18 \
    --wav-scp data/train/wav.scp --output-dir feats/sweep
# -> feats/sweep/layer10/ layer12/ layer14/ layer16/ layer18/
```

Probing those five layers separately costs 10+12+14+16+18 = 70
layer-evaluations. Together it costs 18, plus one model load and one pass over
the audio instead of five. Measured at **3.9x faster** on a 3696-utterance set.

## The layer_norm gotcha

Taking layer *k* by truncating the encoder and reading the last hidden state is
**not** the same as indexing `hidden_states[k]` on the full model.
`Wav2Vec2EncoderStableLayerNorm` (the large models: mms-300m, mms-1b,
wavlm-large, wav2vec2-large, xls-r) and `WhisperEncoder` apply
`encoder.layer_norm` *after* the layer loop, so truncation moves a norm trained
for the top layer onto layer *k*. Base-sized models (hubert-base, wavlm-base,
mHuBERT-147) norm before the loop and are unaffected.

Verified exactly on mms-300m: cosine against `layer_norm(L_k)` is 1.000000 at
layers 8, 14 and 20, while cosine against the raw layer is 0.67 to 0.87.

```bash
--final-layer-norm keep    # default: layer_norm(layer_k)
--final-layer-norm strip   # hidden_states[k], comparable with the literature
```

`keep` is the default because it measures better as a front end: on TIMIT with
mms-300m layer 14 it wins at every HMM-GMM stage by 0.67 to 2.21 PER. The norm
acts as a free per-frame normalisation, which a per-speaker CMVN cannot
substitute for. It also **changes which layer wins** (14 with the norm, 16
without), so a layer study run through a misplaced norm is not measuring the
layer alone. Use `strip` when you need numbers comparable with published layer
studies.

## Notes

- **Frame rate is measured, not assumed.** A short probe pass at load time
  derives seconds-per-frame and writes it to `frame_shift`. Every encoder
  supported here is 20 ms, but hardcoding that would be silently wrong for
  anything else, and a wrong `frame_shift` misaligns everything downstream
  without erroring. Measured rates within 5% of 10, 20 or 40 ms snap to the
  clean value (models typically emit one frame fewer than duration/shift, from
  the convolutional receptive field); anything else is reported with a warning.
  Override with `SSLExtractor(..., frame_shift=0.04)` if you disagree.
- **Whisper** is handled: the decoder is dropped, and because its feature
  extractor pads every utterance to 30 s the frames beyond the real audio are
  discarded. Utterances over 30 s raise rather than being silently truncated.
- **Multi-channel input is rejected**, not mangled. A stereo `(n, 2)` array
  survives `np.squeeze` and the HF feature extractor then reads it as a *batch
  of two*, returning wrong features with no error.
- **Failures abort by default.** A tolerated failure silently shrinks the
  output, which later looks like a smaller test set rather than an error. Raise
  `--max-failures` deliberately.
- **`soundfile`** is only needed for non-wav or non-PCM16 audio:
  `pip install ssl-kaldi-feats[audio]`.

## Where this comes from

Extracted from [ssl-kaldi](https://github.com/ialmajai/ssl-kaldi), a set of
Kaldi recipes that put frozen self-supervised features in front of a classic
HMM-GMM and LF-MMI back end. Measured results, including which choices turned
out not to matter, are in that repo's
[FINDINGS.md](https://github.com/ialmajai/ssl-kaldi/blob/main/FINDINGS.md).

Apache-2.0.
