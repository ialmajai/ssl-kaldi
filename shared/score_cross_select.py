#!/usr/bin/env python3
"""WER with the LM weight and penalty chosen by cross-selection over speakers.

Splits the test speakers into folds (sorted, assigned in turn), scores each fold
at the setting that is best on the other folds, and pools the errors, so no
utterance is scored at a setting chosen on itself. Prints this WER beside the
usual test-best one. Reads the hypotheses a recipe's scoring already wrote:
  <decode>/scoring_kaldi/penalty_*/<lmwt>.txt  with  scoring_kaldi/test_filt.txt
  <decode>/scoring/<lmwt>.tra (word ids)       with  scoring/test_filt.txt
Run from a recipe dir after . ./path.sh (needs compute-wer):
  score_cross_select.py <decode-dir> <utt2spk> [--words words.txt] [--folds 2]
"""
import argparse
import glob
import os
import re
import subprocess
import tempfile
from collections import defaultdict


def read_table(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        parts = line.rstrip("\n").split(None, 1)
        if parts:
            rows[parts[0]] = parts[1] if len(parts) > 1 else ""
    return rows


def settings(decode, words):
    """{setting name: (ref table, hyp table)} for every scored LM weight and penalty."""
    out = {}
    sk = os.path.join(decode, "scoring_kaldi")
    if os.path.isdir(sk):
        ref = read_table(os.path.join(sk, "test_filt.txt"))
        for f in glob.glob(os.path.join(sk, "penalty_*", "*.txt")):
            wip = os.path.basename(os.path.dirname(f))[len("penalty_"):]
            out[f"{os.path.basename(f)[:-4]}_{wip}"] = (ref, read_table(f))
        return out
    sc = os.path.join(decode, "scoring")
    if not words:
        raise SystemExit("scoring/*.tra layout needs --words")
    sym = {}
    for line in open(words, encoding="utf-8"):
        w, i = line.split()
        sym[i] = w
    ref = read_table(os.path.join(sc, "test_filt.txt"))
    for f in glob.glob(os.path.join(sc, "*.tra")):
        hyp = {u: " ".join(w for w in (sym[i] for i in t.split()) if w != "<UNK>")
               for u, t in read_table(f).items()}
        out[os.path.basename(f)[:-4]] = (ref, hyp)
    return out


def errors(ref, hyp, utts):
    """(errors, words) from Kaldi's compute-wer --mode=present on the given utterances."""
    with tempfile.TemporaryDirectory() as tmp:
        r, h = os.path.join(tmp, "ref"), os.path.join(tmp, "hyp")
        with open(r, "w", encoding="utf-8") as fr, open(h, "w", encoding="utf-8") as fh:
            for u in sorted(utts):
                if u in ref:
                    print(u, ref[u], file=fr)
                if u in hyp:
                    print(u, hyp[u], file=fh)
        out = subprocess.run(["compute-wer", "--text", "--mode=present", f"ark:{r}", f"ark:{h}"],
                             capture_output=True, text=True, check=True).stdout
    m = re.search(r"%WER \S+ \[ (\d+) / (\d+)", out)
    return int(m.group(1)), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decode")
    ap.add_argument("utt2spk")
    ap.add_argument("--words", default="")
    ap.add_argument("--folds", type=int, default=2)
    a = ap.parse_args()
    spk = read_table(a.utt2spk)
    cand = settings(a.decode, a.words)
    if not cand:
        raise SystemExit(f"no scored hypotheses under {a.decode}")
    ref_utts = set(next(iter(cand.values()))[0])
    speakers = sorted({spk[u] for u in ref_utts if u in spk})
    fold_of = {s: i % a.folds for i, s in enumerate(speakers)}
    utts = defaultdict(set)
    for u in ref_utts:
        if u in spk:
            utts[fold_of[spk[u]]].add(u)
    # err[setting][fold] = (errors, words)
    err = {s: {k: errors(r, h, utts[k]) for k in utts} for s, (r, h) in cand.items()}
    total = lambda s, ks: sum(err[s][k][0] for k in ks)
    words = sum(err[next(iter(err))][k][1] for k in utts)
    best = min(err, key=lambda s: total(s, utts))
    e_cross, chosen = 0, []
    for k in sorted(utts):
        others = [j for j in utts if j != k]
        s = min(err, key=lambda s: total(s, others))
        chosen.append(s)
        e_cross += err[s][k][0]
    print(f"test-best {100 * total(best, utts) / words:.2f} ({best})  "
          f"cross-selected {100 * e_cross / words:.2f} ({' '.join(chosen)})  "
          f"words={words} speakers={len(speakers)} folds={a.folds}")


if __name__ == "__main__":
    main()
