"""Recover a PCA projection (components, mean) from paired raw / PCA features.
Usage: recover_pca.py RAW_SCP PCA_SCP OUT.pt [TRUE_MODEL.pt]
Fits y = x A + b by least squares on up to 400 utterances, converts to the
pca.py format (y = (x - mean) @ components.T), and reports the fit residual on
held-out utterances. With TRUE_MODEL, also compares against it."""
import sys
import kaldiio
import numpy as np
import torch

raw_scp, pca_scp, out = sys.argv[1:4]
true = sys.argv[4] if len(sys.argv) > 4 else None
raw = dict(kaldiio.load_scp(raw_scp))
pca = dict(kaldiio.load_scp(pca_scp))
utts = sorted(set(raw) & set(pca))
print(f"paired utterances: {len(utts)}")
fit, held = utts[:400], utts[400:600]

def stack(keys):
    X, Y = [], []
    for k in keys:
        x, y = raw[k], pca[k]
        if len(x) != len(y):
            raise SystemExit(f"{k}: {len(x)} raw vs {len(y)} pca frames")
        X.append(x); Y.append(y)
    return np.concatenate(X).astype(np.float64), np.concatenate(Y).astype(np.float64)

X, Y = stack(fit)
X1 = np.hstack([X, np.ones((len(X), 1))])
W, *_ = np.linalg.lstsq(X1, Y, rcond=None)
A, b = W[:-1], W[-1]                     # y = x A + b
components = A.T                          # (30, D)
mean = -b @ np.linalg.pinv(A)             # mean @ A == -b
Xh, Yh = stack(held)
Yp = (Xh - mean) @ components.T
rel = np.abs(Yp - Yh).max() / np.abs(Yh).max()
print(f"held-out frames: {len(Xh)}  max abs err {np.abs(Yp - Yh).max():.4f}  "
      f"relative {rel:.2e}  rms err {np.sqrt(((Yp - Yh) ** 2).mean()):.4f}  rms y {np.sqrt((Yh ** 2).mean()):.3f}")
print(f"row orthonormality: max |C C^T - I| = {np.abs(components @ components.T - np.eye(len(components))).max():.2e}")
if true:
    t = torch.load(true, map_location="cpu", weights_only=True)
    tc, tm = t["components"].double().numpy(), t["mean"].double().numpy()
    print(f"vs true model: max |components diff| {np.abs(components - tc).max():.2e}, "
          f"projected-mean diff {np.abs(mean @ components.T - tm @ tc.T).max():.2e}")
torch.save({"components": torch.from_numpy(components).float(),
            "mean": torch.from_numpy(mean).float()}, out)
print(f"wrote {out}")
