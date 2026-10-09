"""Fused word confidence (lattice MBR + TruCLeS CEM) for one demo utterance.

Runs make_cem_data.sh's steps on a single lattice from the persistent decoder,
builds the CEM input with train_cem.word_inputs, and applies the logistic fusion
fitted by fuse_mbr_cem.py --out. MODEL_DIR holds fusion.pt, pdf2phone.txt and
cem.raw (the chain net with output = [prefinal activations, logits]).
"""
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import kaldiio
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuse_mbr_cem import cem_net, logit  # noqa: E402
from train_cem import phone_pool, word_inputs  # noqa: E402

HIDDEN_DIM = 192  # prefinal-chain.batchnorm2

STEPS = """
. ./path.sh
lex=data/lang/phones/align_lexicon.int
lattice-copy --include=$tmp/utt.list ark:$ark ark:$tmp/lat.ark
lattice-1best ark:$tmp/lat.ark ark:- | lattice-align-words-lexicon $lex $mdl ark:- ark:- | \\
  nbest-to-ctm --frame-shift=1 --print-silence=false ark:- - > $tmp/words.ctm
lattice-1best ark:$tmp/lat.ark ark:- | nbest-to-linear ark:- ark:- | \\
  ali-to-phones --per-frame=true $mdl ark:- ark,t:$tmp/phones.txt
lattice-align-words-lexicon $lex $mdl ark:$tmp/lat.ark ark:- | \\
  lattice-to-ctm-conf --decode-mbr=true --frame-shift=1 ark:- - > $tmp/mbr.ctm
nnet3-compute --frame-subsampling-factor=2 --frames-per-chunk=150 --extra-left-context-initial=0 \\
  --extra-right-context-final=0 --use-gpu=no $net ark:$tmp/feats.ark ark:$tmp/out.ark
"""


class FusedConfidence:
    def __init__(self, model_dir, chain_dir, recipe_root):
        self.model_dir, self.chain_dir, self.root = Path(model_dir), Path(chain_dir), Path(recipe_root)
        st = torch.load(self.model_dir / "fusion.pt", map_location="cpu", weights_only=True)
        self.cem = cem_net(len(st["mu"]))
        self.cem.load_state_dict(st["cem"])
        self.cem.eval()
        self.mu, self.sd = st["mu"].numpy(), st["sd"].numpy()
        self.temp, self.coef, self.bias = st["temp"], st["coef"].numpy(), st["bias"]
        self.pool, self.ph_idx = phone_pool(self.model_dir / "pdf2phone.txt")
        lang = self.root / "data/lang"
        self.phones = {int(l.split()[1]): l.split()[0] for l in open(lang / "phones.txt")}
        self.words = {int(l.split()[1]): l.split()[0] for l in open(lang / "words.txt")}

    def __call__(self, lat_ark, utt_id, feats, cmvn_feats):
        """feats: the upsampled input features; cmvn_feats: what the decoder saw.
        Returns [(word, confidence)] in sentence order, or None when the MBR and
        1-best word sequences differ (the fusion pairs MBR and 1-best words by position)."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "utt.list").write_text(utt_id + "\n")
            kaldiio.save_ark(f"{tmp}/feats.ark", {utt_id: cmvn_feats.astype(np.float32)})
            env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in
                           (("tmp", tmp), ("ark", lat_ark), ("mdl", self.chain_dir / "final.mdl"),
                            ("net", self.model_dir / "cem.raw")))
            subprocess.run(["bash", "-c", f"set -eo pipefail; {env}; {STEPS}"], cwd=str(self.root),
                           check=True, capture_output=True, text=True)
            words = [(self.words[int(l.split()[4])], int(float(l.split()[2])),
                      int(float(l.split()[2])) + int(float(l.split()[3])))
                     for l in open(f"{tmp}/words.ctm")]
            mbr = [(self.words[int(l.split()[4])], float(l.split()[5])) for l in open(f"{tmp}/mbr.ctm")]
            ali = [self.phones[int(x)] for x in open(f"{tmp}/phones.txt").read().split()[1:]]
            out = dict(kaldiio.load_ark(f"{tmp}/out.ark"))[utt_id]
        if [w for w, _, _ in words] != [w for w, _ in mbr]:
            return None
        if not words:
            return []
        _, inputs = word_inputs(feats, out[:, HIDDEN_DIM:], out[:, :HIDDEN_DIM], words, ali,
                                self.temp, self.pool, self.ph_idx)
        X = np.stack([f for f, _, _ in inputs]).astype(np.float32)
        with torch.no_grad():
            cem = self.cem(torch.tensor((X - self.mu) / self.sd)).squeeze(1).numpy()
        cp = np.array([c for _, _, c in inputs])
        m = np.array([c for _, c in mbr])
        z = np.stack([logit(m), logit(cem), logit(cp)], 1) @ self.coef + self.bias
        return [(w, float(c)) for (w, _), c in zip(mbr, 1 / (1 + np.exp(-z)))]
