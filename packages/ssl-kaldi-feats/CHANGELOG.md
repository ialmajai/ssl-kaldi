# Changelog

## Unreleased

- Open: verify against transformers 5.x and lift the `<5` pin if the fixture
  still reproduces. Run `pytest -m needs_model tests/` on a 5.x environment.

## 0.1.2 (2026-10-15)

- Unsigned integer PCM (for example 8-bit wav as returned by
  `scipy.io.wavfile`) is centred on its midpoint before scaling, so silence
  maps to 0.0. 0.1.1 divided by the dtype maximum only, which mapped silence
  to 0.5 and shifted the features without an error. Signed PCM is unchanged.
- Releases are published from GitHub Actions through PyPI Trusted Publishing.

## 0.1.1 (2026-10-13)

- Piped wav input that is not 16-bit PCM (24/32-bit, float) is decoded through
  soundfile. 0.1.0 read every piped wav as int16, so such input produced
  wrong features without an error.
- An unreadable file or failing pipe in `wav.scp` or a file list now counts as
  one failure against `--max-failures`. 0.1.0 read audio outside the
  per-utterance error handling, so one bad entry ended the job.
- Compatibility test tolerance is now relative to each fixture's own scale,
  with a separate cosine check. The absolute bound reported a false failure on
  transformers 4.57.6 because `strip` features carry roughly 20x the standard
  deviation of `keep` ones. Test-only: the shipped code in 0.1.0 is unaffected,
  and the fix missed that release by minutes.

## 0.1.0 (2026-08-05)

First release.

- Frame-level features from any HuggingFace speech encoder, written as Kaldi
  `ark`/`scp` with `utt2num_frames`, `utt2dur` and `frame_shift`. No Kaldi
  installation required.
- `--layers` serves several layers from one forward pass to the deepest,
  measured 3.9x faster than one pass per layer on a 3696-utterance set.
- `--final-layer-norm keep|strip` handles the encoder-final `layer_norm` that
  truncation otherwise moves onto the requested layer. Affects
  `do_stable_layer_norm` models and Whisper; changes which layer measures best,
  so it is not cosmetic.
- Whisper support: decoder dropped, 30 s padding frames discarded, utterances
  over 30 s raise rather than being silently truncated.
- Frame shift is measured from a probe pass rather than assumed, and snapped to
  10, 20 or 40 ms when within tolerance.
- Multi-channel audio is rejected rather than mangled: a stereo array survives
  `np.squeeze` and the HF feature extractor then reads it as a batch of two.
- `transformers` pinned `>=4.46,<5`. The package depends on encoder internals,
  and features are only stable to about 1e-3 relative even across 4.x minor
  versions, so a major release is a real risk until tested.
