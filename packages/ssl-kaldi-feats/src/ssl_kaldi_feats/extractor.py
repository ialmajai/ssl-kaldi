"""Frame-level features from any HuggingFace speech encoder.

The awkward parts of this, and why they are here rather than in your own script:

* **The encoder-final layer_norm.** Taking layer k by truncating the encoder and
  reading the last hidden state is NOT the same as indexing hidden_states[k].
  Wav2Vec2EncoderStableLayerNorm (the large models) and WhisperEncoder apply
  layer_norm AFTER the layer loop, so truncation moves a norm trained for the
  top layer onto layer k. See `final_layer_norm`.
* **Several layers from one pass.** Probing layers 10,12,14,16,18 separately
  costs 70 layer-evaluations; together it costs 18. Doing that correctly needs
  the norm applied per captured layer, not once at the end.
* **Whisper** pads every utterance to 30 s, so the encoder always returns 1500
  frames and a 3 s utterance would otherwise carry 1350 frames of padding.
"""
import logging

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel

logger = logging.getLogger(__name__)
# Set the level on our own logger. AutoModel.from_pretrained() resets the ROOT
# logger to WARNING, which silently swallows every INFO emitted afterwards.
logger.setLevel(logging.INFO)

EXPECTED_SAMPLE_RATE = 16000
WHISPER_WINDOW_SECONDS = 30
WHISPER_SAMPLES_PER_FRAME = 320


def preprocess_waveform(waveform):
    """Scale to float32 in [-1, 1]. Rejects multi-channel rather than mangle it.

    np.squeeze leaves an (n, 2) array untouched and the HF feature extractor
    then reads it as a BATCH of two utterances, returning wrong features with
    no error at all.
    """
    waveform = np.asarray(waveform)
    if waveform.ndim > 1 and min(waveform.shape) > 1:
        raise ValueError(
            f"expected mono audio, got shape {waveform.shape}. Downmix or "
            "select a channel first"
        )
    if waveform.ndim > 1:
        waveform = np.squeeze(waveform)
    # np.squeeze turns a one-sample array into a 0-d scalar, which then fails
    # on len() deep inside the feature extractor.
    waveform = np.atleast_1d(waveform)
    if np.issubdtype(waveform.dtype, np.integer):
        return waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    return waveform.astype(np.float32)


class SSLExtractor:
    """Wraps an HF speech encoder as a fixed frame-level feature extractor.

    Parameters
    ----------
    model_id : str
        Any HuggingFace model whose encoder exposes ``encoder.layers``, e.g.
        ``facebook/hubert-base-ls960``, ``microsoft/wavlm-large``,
        ``facebook/mms-1b``, ``openai/whisper-large-v3``.
    layers : int or sequence of int
        Layer(s) to return, 1-indexed. Several layers are served from one
        forward pass to the deepest one.
    final_layer_norm : {"keep", "strip"}
        What to do with the encoder-final layer_norm on models that apply it
        after the layer loop. ``keep`` returns ``layer_norm(layer_k)``;
        ``strip`` returns ``hidden_states[k]``, matching what published layer
        studies mean. No effect on models that norm before the loop
        (hubert-base, wavlm-base, mHuBERT-147).

        ``keep`` is the default because it measures better as a front end for
        a PCA plus diagonal-covariance GMM pipeline: on TIMIT with mms-300m
        layer 14 it wins at every GMM stage by 0.67 to 2.21 PER. Use ``strip``
        when you need comparability with the literature.
    device : str, optional
        Defaults to cuda when available.
    frame_shift : float, optional
        Seconds per output frame. Measured from a probe forward pass when not
        given, which is the safe default: hardcoding 20 ms is correct for every
        encoder this package currently supports and silently wrong for anything
        else. Pass a value only to override a measurement you disagree with.
    """

    def __init__(self, model_id, layers, final_layer_norm="keep", device=None,
                 frame_shift=None):
        if final_layer_norm not in ("keep", "strip"):
            raise ValueError("final_layer_norm must be 'keep' or 'strip'")
        self.model_id = model_id
        self.layers = [layers] if isinstance(layers, int) else sorted(set(layers))
        self.final_layer_norm_mode = final_layer_norm

        logger.info("loading %s", model_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()

        self.is_whisper = type(self.model).__name__.startswith("Whisper")
        if not hasattr(self.model, "encoder") or not hasattr(self.model.encoder, "layers"):
            raise ValueError(
                f"{model_id} has no encoder.layers; this tool supports encoders "
                "of the wav2vec2/HuBERT/WavLM/Whisper family"
            )
        self.num_layers = len(self.model.encoder.layers)
        for layer in self.layers:
            if not 1 <= layer <= self.num_layers:
                raise ValueError(
                    f"layer {layer} out of range 1..{self.num_layers} for {model_id}"
                )

        # One pass only needs to reach the deepest requested layer.
        self.model.encoder.layers = self.model.encoder.layers[: max(self.layers)]

        norms_after_loop = self.is_whisper or getattr(
            self.model.config, "do_stable_layer_norm", False
        )
        # Take the norm out of the graph either way and re-apply per layer when
        # keeping. A single pass would otherwise norm ONLY the deepest layer,
        # silently mixing keep and strip features inside one sweep.
        self._final_ln = None
        if norms_after_loop:
            if final_layer_norm == "keep":
                self._final_ln = self.model.encoder.layer_norm
            self.model.encoder.layer_norm = torch.nn.Identity()
            logger.info(
                "encoder norms after the layer loop; final_layer_norm=%s",
                final_layer_norm,
            )

        if self.is_whisper:
            self.model = self.model.encoder

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self._final_ln is not None:
            self._final_ln = self._final_ln.to(self.device)

        # Measured, not assumed. Costs one short forward pass at load time.
        self.frame_shift = (frame_shift if frame_shift is not None
                            else self._measure_frame_shift())

    # Rates a Kaldi pipeline is likely to expect. A measured value within
    # tolerance of one of these is snapped to it, so a model that emits
    # 199 frames for 4 s (one short, from input stacking) still reports a
    # clean 0.02 rather than 0.0201.
    _KNOWN_FRAME_SHIFTS = (0.01, 0.02, 0.04)
    _SNAP_TOLERANCE = 0.05  # 5 per cent

    def _measure_frame_shift(self, seconds=4.0):
        """Derive seconds-per-frame from an actual forward pass.

        Hardcoding 0.02 is right for every wav2vec2/HuBERT/WavLM/Whisper
        encoder, and silently wrong for anything else. Measuring means a model
        at a different rate is reported correctly instead of writing a
        frame_shift that quietly misaligns every downstream alignment.
        """
        probe = np.zeros(int(seconds * EXPECTED_SAMPLE_RATE), dtype=np.float32)
        frames = len(self.extract(probe)[self.layers[0]])
        if frames < 1:
            raise ValueError(f"{self.model_id} produced no frames for a probe")
        measured = seconds / frames
        for known in self._KNOWN_FRAME_SHIFTS:
            if abs(measured - known) / known <= self._SNAP_TOLERANCE:
                if measured != known:
                    logger.info(
                        "measured frame shift %.5fs, reporting %.2fs "
                        "(%d frames for %.1fs)", measured, known, frames, seconds
                    )
                return known
        logger.warning(
            "%s emits %d frames for %.1fs, i.e. %.5f s/frame, which is not a "
            "rate this package has been tested against. Check that downstream "
            "alignment expects it.", self.model_id, frames, seconds, measured
        )
        return measured

    def extract(self, waveform, sample_rate=EXPECTED_SAMPLE_RATE):
        """Return ``{layer: ndarray of shape (frames, dim)}``."""
        if sample_rate != EXPECTED_SAMPLE_RATE:
            raise ValueError(
                f"sample rate {sample_rate} != {EXPECTED_SAMPLE_RATE}; resample first"
            )
        waveform = preprocess_waveform(waveform)

        if self.is_whisper and len(waveform) > WHISPER_WINDOW_SECONDS * sample_rate:
            raise ValueError(
                f"utterance is {len(waveform) / sample_rate:.1f}s; Whisper "
                f"truncates above {WHISPER_WINDOW_SECONDS}s, which would "
                "silently discard audio. Split it first"
            )

        inputs = self.feature_extractor(
            waveform, sampling_rate=sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states

        out = {}
        for layer in self.layers:
            feats = hidden[layer]
            if self._final_ln is not None:
                with torch.no_grad():
                    feats = self._final_ln(feats)
            feats = feats.squeeze(0).cpu().numpy()
            if self.is_whisper:
                feats = feats[: len(waveform) // WHISPER_SAMPLES_PER_FRAME]
            out[layer] = feats
        return out

    def extract_one(self, waveform, sample_rate=EXPECTED_SAMPLE_RATE):
        """Single-layer convenience wrapper."""
        return self.extract(waveform, sample_rate)[self.layers[0]]
