#!/usr/bin/env python3
"""Regenerate the compatibility fixture. Run deliberately, on a version you
trust, never just to turn a red test green."""
import json
import os

import numpy as np
import torch
import transformers

from ssl_kaldi_feats import SSLExtractor

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)
torch.manual_seed(0)
wav = (torch.randn(16000) * 0.05).numpy().astype(np.float32)
np.save(os.path.join(DATA, "ref_input.npy"), wav)

ref, meta = {}, {"transformers": transformers.__version__, "torch": torch.__version__}
for mid, layer, mode, tag in [
    ("facebook/hubert-base-ls960", 6, "keep", "base_keep"),
    ("facebook/mms-300m", 14, "keep", "large_keep"),
    ("facebook/mms-300m", 14, "strip", "large_strip"),
]:
    ex = SSLExtractor(mid, layer, mode, device="cpu")
    ref[tag] = ex.extract_one(wav)
    meta[tag] = {"model": mid, "layer": layer, "mode": mode,
                 "shape": list(ref[tag].shape),
                 "checksum": float(np.abs(ref[tag]).sum())}
np.savez_compressed(os.path.join(DATA, "reference_feats.npz"), **ref)
json.dump(meta, open(os.path.join(DATA, "reference_meta.json"), "w"), indent=2)
print(f"regenerated on transformers {transformers.__version__}")
