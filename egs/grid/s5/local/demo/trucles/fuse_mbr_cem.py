#!/usr/bin/env python3
"""Late fusion of lattice MBR confidence and the TruCLeS CEM output.

A logistic regression on [logit(MBR), logit(CEM), logit(class prob)] is fitted
to word correctness on the training split. CEM scores for the training split
are out-of-fold (5 folds by utterance), so the fusion never sees in-sample CEM
outputs. Inputs and splits as train_cem.py.
  fuse_mbr_cem.py --train SET... --eval SET... [--temp 0.05] [--out DIR]
--out saves fusion.pt (CEM weights, input normalisation, temp, fusion weights) and
the pdf2phone.txt it was trained with, for the demo's local/demo/trucles/score.py.
"""
import argparse
import os
import shutil
import sys
import zlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_cem import load_set, metrics, shrinkage_loss  # noqa: E402


def cem_net(dim):
    return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2),
                         nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())


def fit_cem(X, y, epochs=40, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-5
    m = cem_net(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yt = torch.tensor((X - mu) / sd), torch.tensor(y)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = shrinkage_loss(m(Xt[b]).squeeze(1), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    predict = lambda Z: m(torch.tensor((Z - mu) / sd)).squeeze(1).detach().numpy()
    predict.state = {"cem": m.state_dict(), "mu": torch.tensor(mu), "sd": torch.tensor(sd)}
    return predict


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--out")
    a = ap.parse_args()
    lexicon = {}
    for l in open("data/lang/phones/align_lexicon.txt"):
        f = l.split()
        lexicon.setdefault(f[0], f[2:])
    tr = [load_set(s, lexicon, a.temp) for s in a.train]
    X, y, c, cp, mbr = (np.concatenate([t[i] for t in tr]) for i in range(5))
    utts = np.array([k[0] for t in tr for k in t[5]])
    fold = np.array([zlib.crc32(u.encode()) % 5 for u in utts])
    oof = np.zeros(len(y), np.float32)
    for f in range(5):
        oof[fold == f] = fit_cem(X[fold != f], y[fold != f])(X[fold == f])
    ok = ~np.isnan(mbr)
    F = lambda m, e, p: np.stack([logit(m), logit(e), logit(p)], 1)
    fus = LogisticRegression(C=1.0).fit(F(mbr[ok], oof[ok], cp[ok]), c[ok])
    print("fusion weights [mbr, cem, class_prob]:", np.round(fus.coef_[0], 3), "bias", round(fus.intercept_[0], 3))
    cem = fit_cem(X, y)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        torch.save(dict(cem.state, temp=a.temp, coef=torch.tensor(fus.coef_[0]),
                        bias=float(fus.intercept_[0])), f"{a.out}/fusion.pt")
        shutil.copy(f"{a.train[0].partition(':')[0]}/pdf2phone.txt", a.out)
    mbr_only = LogisticRegression().fit(logit(mbr[ok])[:, None], c[ok])
    for spec in a.eval:
        Xe, _, ce, be, me, _ = load_set(spec, lexicon, a.temp)
        ok = ~np.isnan(me)
        pe = cem(Xe)[ok]
        print(spec)
        for name, conf in (("mbr+platt", mbr_only.predict_proba(logit(me[ok])[:, None])[:, 1]),
                           ("fusion", fus.predict_proba(F(me[ok], pe, be[ok]))[:, 1])):
            r = metrics(conf, ce[ok])
            print(f"  {name:10s} auc {r['auc']:.3f} nce {r['nce']:+.3f} ece {r['ece']:.3f}")


if __name__ == "__main__":
    main()
