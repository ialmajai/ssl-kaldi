#!/usr/bin/env python3
"""TruCLeS word-confidence model (Ravi et al., Interspeech 2025) for the Kaldi chain.

Per 1-best word: input is the mean over its frames of the AV-HuBERT input features,
the prefinal hidden layer and the output softmax (the paper's [a, h, s], eqs 5-7).
Target is TruCLeS (eqs 2-3): the mean over the hyp word's phones of the net's
probability of the aligned reference phone (0 for an inserted phone), times the
letter-level Levenshtein similarity of hyp and reference word; 0 for an inserted
word. The model is an MLP with a sigmoid output, trained with shrinkage loss.

Phone probabilities are the chain softmax (logits / T) summed over each phone's
pdfs. Chain outputs are not frame posteriors and their softmax is nearly flat, so
T is fitted on the training split to minimise the class-prob baseline's ECE.

  train_cem.py --train SET[:SPK_FILTER] ... --eval SET[:SPK_FILTER] ... --out DIR
SET is a make_cem_data.sh dir with a 'data' link to its Kaldi data dir.
SPK_FILTER is '+file' (keep speakers in file) or '-file' (drop them) or '~s1,s2'.
"""
import argparse
import json
import os

import kaldiio
import numpy as np
import torch
from torch import nn


def edit_align(a, b):
    """Levenshtein alignment; returns per-a-item index into b (None = insertion)."""
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    out, i, j = [None] * n, n, m
    while i > 0 and j > 0:
        if d[i, j] == d[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return out


def lev_sim(a, b):
    if not a and not b:
        return 1.0
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return 1.0 - d[len(b)] / max(len(a), len(b))


def runs(seq):
    """Collapse per-frame phones into (phone, start, end) segments, dropping SIL."""
    out, s = [], 0
    for t in range(1, len(seq) + 1):
        if t == len(seq) or seq[t] != seq[s]:
            if seq[s] != "SIL":
                out.append((seq[s], s, t))
            s = t
    return out


def speakers_ok(spk, filt):
    if not filt:
        return True
    if filt[0] == "~":
        return spk in filt[1:].split(",")
    keep = {l.split()[0] for l in open(filt[1:]) if l.strip()}
    return (spk in keep) == (filt[0] == "+")


def phone_pool(pdf2phone):
    """Read a pdf2phone.txt; return the (pdfs, phones) summing matrix and phone index."""
    pdf2ph = [p for _, p in sorted((int(l.split()[0]), l.split()[1]) for l in open(pdf2phone))]
    ph_idx = {p: i for i, p in enumerate(sorted(set(pdf2ph)))}
    pool = np.zeros((len(pdf2ph), len(ph_idx)), dtype=np.float32)
    for k, p in enumerate(pdf2ph):
        pool[k, ph_idx[p]] = 1.0
    return pool, ph_idx


def word_inputs(x, lg, hid, words, ali, temp, pool, ph_idx):
    """Build each 1-best word's CEM input and class-prob baseline for one utterance.
    x: input features (any multiple of the net's frame rate); lg, hid: net output
    and prefinal activations; words: [(word, start, end)] in net frames; ali: the
    1-best phone name per net frame. Returns (phone posteriors, [(feat, segs, cp)])."""
    lg = lg / temp
    sm = np.exp(lg - lg.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    php = sm @ pool                                       # (T, phones)
    x = (x - x.mean(0)) / (x.std(0) + 1e-5)               # utterance CMVN, as the demo
    fsf = x.shape[0] / lg.shape[0]
    out = []
    for w, s, e in words:
        e = max(e, s + 1)
        a = x[int(s * fsf):max(int(e * fsf), int(s * fsf) + 1)].mean(0)
        feat = np.concatenate([a, hid[s:e].mean(0), sm[s:e].mean(0)])
        segs = runs(ali[s:e])
        cp = float(np.mean([php[s + a0:s + a1, ph_idx[p]].mean() for p, a0, a1 in segs])) if segs else 0.0
        out.append((feat, segs, cp))
    return php, out


def load_set(spec, lexicon, temp=1.0, with_mbr=False):
    d, _, filt = spec.partition(":")
    data = os.path.realpath(os.path.join(d, "data"))
    phones = {int(l.split()[1]): l.split()[0] for l in open("data/lang/phones.txt")}
    pool, ph_idx = phone_pool(f"{d}/pdf2phone.txt")
    utt2spk = dict(l.split() for l in open(f"{data}/utt2spk"))
    ref = {l.split()[0]: l.split()[1:] for l in open(f"{d}/text")}
    ali = {l.split()[0]: [phones[int(x)] for x in l.split()[1:]] for l in open(f"{d}/phones.txt")}
    ctm = {}
    for l in open(f"{d}/words.ctm"):
        u, _, s, n, w = l.split()[:5]
        ctm.setdefault(u, []).append((w, int(float(s)), int(float(s)) + int(float(n))))
    logits = dict(kaldiio.load_ark(f"{d}/logits.ark"))
    hidden = dict(kaldiio.load_ark(f"{d}/hidden.ark"))
    feats = kaldiio.load_scp(f"{data}/feats.scp")
    mbr = {}
    if os.path.exists(f"{d}/mbr.ctm"):
        for l in open(f"{d}/mbr.ctm"):
            mbr.setdefault(l.split()[0], []).append(float(l.split()[5]))
    X, y, correct, base, mconf, keys = [], [], [], [], [], []
    for u, words in ctm.items():
        if not speakers_ok(utt2spk[u], filt) or u not in logits:
            continue
        php, inputs = word_inputs(feats[u], logits[u], hidden[u], words, ali[u], temp, pool, ph_idx)
        hyp = [w for w, _, _ in words]
        wmap = edit_align(hyp, ref[u])
        for k, ((w, s, _), (feat, segs, cp)) in enumerate(zip(words, inputs)):
            hyp_ph = [p for p, _, _ in segs]
            j = wmap[k]
            if j is None:
                c = 0.0
            else:
                rw = ref[u][j]
                ref_ph = lexicon.get(rw, [])
                pmap = edit_align(hyp_ph, ref_ph)
                eta = [0.0 if jj is None or ref_ph[jj] not in ph_idx else
                       float(php[s + a0:s + a1, ph_idx[ref_ph[jj]]].mean())
                       for (p, a0, a1), jj in zip(segs, pmap)]
                c = (np.mean(eta) if eta else 0.0) * lev_sim(w, rw)
            m = mbr.get(u, [])
            mc = m[k] if k < len(m) else 1.0
            if with_mbr:
                feat = np.concatenate([feat, [mc, cp]])
            X.append(feat)
            y.append(c)
            correct.append(int(j is not None and w == ref[u][j]))
            base.append(cp)                               # the paper's "Class Prob" baseline
            mconf.append(mc if k < len(m) else np.nan)    # the demo's current lattice MBR
            keys.append((u, k, w))
    return (np.array(X, np.float32), np.array(y, np.float32), np.array(correct),
            np.array(base, np.float32), np.array(mconf, np.float32), keys)


def shrinkage_loss(pred, tgt, a=10.0, c=0.2):
    l = (pred - tgt).abs()
    return (l ** 2 * torch.exp(tgt) / (1 + torch.exp(a * (c - l)))).mean()


def metrics(conf, correct):
    from sklearn.metrics import roc_auc_score
    conf = np.clip(conf, 1e-6, 1 - 1e-6)
    p = correct.mean()
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    hc = -np.mean(correct * np.log2(conf) + (1 - correct) * np.log2(1 - conf))
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(conf[bins == b].mean() - correct[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    return {"auc": roc_auc_score(correct, conf) if 0 < p < 1 else float("nan"),
            "nce": (h - hc) / h, "ece": ece, "n": int(len(correct)), "err_rate": float(1 - p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, help="softmax temperature; fitted on --train if unset")
    ap.add_argument("--with-mbr", action="store_true", help="add lattice MBR and class prob as inputs")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    lexicon = {}
    for l in open("data/lang/phones/align_lexicon.txt"):
        f = l.split()
        lexicon.setdefault(f[0], f[2:])
    if a.temp is None:
        best = None
        for t in (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05):
            tr = [load_set(s, lexicon, t, a.with_mbr) for s in a.train]
            e = metrics(np.concatenate([x[3] for x in tr]), np.concatenate([x[2] for x in tr]))["ece"]
            print(f"temp {t}: class-prob ECE on train {e:.4f}", flush=True)
            if best is None or e < best[0]:
                best = (e, t, tr)
        _, a.temp, tr = best
    else:
        tr = [load_set(s, lexicon, a.temp, a.with_mbr) for s in a.train]
    print(f"using temp {a.temp}", flush=True)
    X = np.concatenate([t[0] for t in tr])
    y = np.concatenate([t[1] for t in tr])
    mu, sd = X.mean(0), X.std(0) + 1e-5
    model = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Dropout(0.2),
                          nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = shrinkage_loss(model(Xt[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 10 == 9:
            print(f"epoch {ep + 1} loss {tot / len(Xt):.5f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "dim": X.shape[1], "temp": a.temp, "with_mbr": a.with_mbr}, f"{a.out}/cem.pt")
    model.eval()
    # Fit an isotonic map per method on the training split, as the demo's
    # existing calibration does, so every method is compared calibrated.
    from sklearn.isotonic import IsotonicRegression
    with torch.no_grad():
        ptr = model(Xt).squeeze(1).numpy()
    ctr = np.concatenate([t[2] for t in tr])
    cal = {}
    for name, raw in (("cem", ptr), ("cp", np.concatenate([t[3] for t in tr])),
                      ("mbr", np.concatenate([t[4] for t in tr]))):
        ok = ~np.isnan(raw)
        cal[name] = IsotonicRegression(y_min=1e-3, y_max=1 - 1e-3, out_of_bounds="clip").fit(raw[ok], ctr[ok])
    import pickle
    pickle.dump(cal["cem"], open(f"{a.out}/cem_isotonic.pkl", "wb"))
    res = {}
    for spec in a.eval:
        Xe, ye, ce, be, me, _ = load_set(spec, lexicon, a.temp, a.with_mbr)
        with torch.no_grad():
            pe = model(torch.tensor((Xe - mu) / sd)).squeeze(1).numpy()
        res[spec] = {"cem_target_mae": float(np.abs(pe - ye).mean())}
        for name, raw, f in (("trucles_cem", pe, cal["cem"]), ("class_prob", be, cal["cp"]),
                             ("lattice_mbr", me, cal["mbr"])):
            ok = ~np.isnan(raw)
            res[spec][name] = metrics(raw[ok], ce[ok])
            res[spec][name + "+iso"] = metrics(f.predict(raw[ok]), ce[ok])
        print(spec, json.dumps(res[spec], indent=1), flush=True)
    json.dump(res, open(f"{a.out}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
