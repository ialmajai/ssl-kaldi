#!/usr/bin/env python3
"""Interpolate an exported fine-tuned AV-HuBERT encoder toward its pretrained weights.

Writes (1 - alpha) * PRETRAINED + alpha * FINETUNED for every floating-point tensor
of the model; integer tensors (BatchNorm counters) come from PRETRAINED. Both inputs
are plain checkpoints with identical keys (export_lora_encoder.py output and the
checkpoint it was exported against), so for a merged LoRA encoder this moves exactly
alpha of the way along the adapter update. cfg and task_state come from PRETRAINED,
so compute_avhubert_feats.py loads the result as it loads base_vox_iter5.pt.
  interpolate_encoder.py PRETRAINED.pt FINETUNED.pt ALPHA OUT.pt
"""
import sys

import torch


def main():
    pre_path, ft_path, alpha, out_path = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    pre = torch.load(pre_path, map_location="cpu")
    ft = torch.load(ft_path, map_location="cpu")
    pm, fm = pre["model"], ft["model"]
    if pm.keys() != fm.keys():
        sys.exit(f"key mismatch: {sorted(pm.keys() ^ fm.keys())[:5]}")
    mixed, moved = {}, 0
    for k, p in pm.items():
        f = fm[k]
        if p.shape != f.shape:
            sys.exit(f"shape mismatch at {k}: {tuple(p.shape)} vs {tuple(f.shape)}")
        if p.is_floating_point():
            mixed[k] = ((1 - alpha) * p.float() + alpha * f.float()).to(p.dtype)
            moved += int(not torch.equal(p, f))
        else:
            mixed[k] = p
    out = dict(pre)
    out["model"] = mixed
    out["interpolation"] = {"pretrained": pre_path, "finetuned": ft_path, "alpha": alpha}
    torch.save(out, out_path)
    print(f"alpha={alpha}: {moved} of {len(pm)} tensors differ from pretrained -> {out_path}")


if __name__ == "__main__":
    main()
