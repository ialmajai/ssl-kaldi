"""Guards the transformers internals this package depends on.

The extractor does structural surgery that is not public API:

* truncates ``model.encoder.layers``
* reads ``config.do_stable_layer_norm``
* replaces ``model.encoder.layer_norm`` with Identity

A transformers release can change any of those. The dangerous failure is not an
exception, it is **silently different features**, so these tests compare against
a fixture generated on a known-good version rather than merely checking that
the code runs.

Downloads models, so it is opt-in:

    pytest -m needs_model tests/

Regenerate the fixture deliberately, never to make a red test go green:

    python tests/make_reference.py
"""
import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.needs_model

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REF = os.path.join(DATA, "reference_feats.npz")
META = os.path.join(DATA, "reference_meta.json")


@pytest.fixture(scope="module")
def reference():
    if not os.path.exists(REF):
        pytest.skip("reference fixture not present")
    return np.load(REF), json.load(open(META))


# Features are NOT bit-reproducible across transformers minor versions: kernel
# and accumulation-order changes move them by about 1e-3 relative. Measured
# 4.46.3 against 4.57.6, where cosine similarity stayed 1.0 to nine decimals
# while the largest absolute difference was 4.3e-3 on features with std 2.58.
#
# The tolerance is therefore RELATIVE to each case's own scale. An absolute
# bound cannot serve both modes here: `strip` features have roughly 20x the
# standard deviation of `keep` ones, so a threshold tight enough for one is
# meaningless for the other.
RELATIVE_TOLERANCE = 1e-2   # ~6x the observed 4.46 -> 4.57 drift
COSINE_TOLERANCE = 1e-6     # direction must be preserved essentially exactly


def test_reference_features_are_reproduced(reference):
    """Same features to within float noise, or the internals moved under us."""
    from ssl_kaldi_feats import SSLExtractor

    ref, meta = reference
    wav = np.load(os.path.join(DATA, "ref_input.npy"))
    import transformers

    problems = []
    for tag in ref.files:
        spec = meta[tag]
        ex = SSLExtractor(spec["model"], spec["layer"], spec["mode"], device="cpu")
        got = ex.extract_one(wav)
        expected = ref[tag]
        if got.shape != tuple(spec["shape"]):
            problems.append(f"{tag}: shape {got.shape} != {tuple(spec['shape'])}")
            continue
        scale = float(expected.std()) or 1.0
        rel = float(np.abs(got - expected).max()) / scale
        cos = float(
            (got * expected).sum()
            / (np.linalg.norm(got) * np.linalg.norm(expected))
        )
        if rel > RELATIVE_TOLERANCE:
            problems.append(f"{tag}: relative maxdiff {rel:.2e}")
        # Catches the case that matters: a change of substance rather than of
        # arithmetic. Float noise leaves direction untouched; a reordered or
        # renormalised representation does not.
        if abs(cos - 1.0) > COSINE_TOLERANCE:
            problems.append(f"{tag}: cosine {cos:.9f}")
    assert not problems, (
        f"features differ from the fixture generated on transformers "
        f"{meta['transformers']} (running {transformers.__version__}): "
        + "; ".join(problems)
    )


def test_stable_layer_norm_flag_still_exists():
    """`do_stable_layer_norm` is how we tell the two encoder shapes apart."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("facebook/mms-300m")
    assert getattr(cfg, "do_stable_layer_norm", None) is True


def test_encoder_layers_is_sliceable():
    """Truncation is what makes a multi-layer sweep cheap."""
    from transformers import AutoModel

    m = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    n = len(m.encoder.layers)
    m.encoder.layers = m.encoder.layers[:3]
    assert n > 3 and len(m.encoder.layers) == 3


def test_hidden_states_indexing_convention():
    """hidden_states[k] must be the output of layer k, with [0] the embeddings."""
    import torch
    from transformers import AutoModel

    m = AutoModel.from_pretrained("facebook/hubert-base-ls960")
    m.eval()
    with torch.no_grad():
        hs = m(torch.zeros(1, 16000), output_hidden_states=True).hidden_states
    assert len(hs) == len(m.encoder.layers) + 1
