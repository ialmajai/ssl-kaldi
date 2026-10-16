#!/usr/bin/env python3
"""Merge a LoRA CTC fine-tune (avh_lora plugin) into a plain AV-HuBERT checkpoint.

Each LoRA weight becomes W + B A * alpha/r; every other encoder tensor is copied.
The result keeps the pretrained checkpoint's cfg and task_state, so
compute_avhubert_feats.py loads it exactly as it loads base_vox_iter5.pt. A full
fine-tune (no LoRA weights) exports the same way.

The CTC head, the fine-tune's cfg and the model's key list go to a small
OUT.ctc_head.pt beside it, so decode_ctc.py --head can rebuild the fine-tuned
model from the two files after the 1.2 GB training checkpoint is deleted.
  export_lora_encoder.py FINETUNED.pt PRETRAINED.pt OUT.pt
"""
import sys

import torch


def main():
    ft_path, pre_path, out_path = sys.argv[1:4]
    ft = torch.load(ft_path, map_location="cpu")
    pre = torch.load(pre_path, map_location="cpu")
    mcfg = ft["cfg"]["model"]
    scale = mcfg["lora_alpha"] / mcfg["lora_rank"]
    src = {k[len("w2v_encoder.w2v_model."):]: v for k, v in ft["model"].items()
           if k.startswith("w2v_encoder.w2v_model.")}
    merged, n_lora, n_copy = {}, 0, 0
    for k, v in src.items():
        if k.endswith(".parametrizations.weight.original"):
            base = k[: -len(".parametrizations.weight.original")]
            A = src[base + ".parametrizations.weight.0.A"].float()
            B = src[base + ".parametrizations.weight.0.B"].float()
            merged[base + ".weight"] = (v.float() + B @ A * scale).to(v.dtype)
            n_lora += 1
        elif ".parametrizations." not in k and k != "mask_emb":  # mask_emb is re-initialised, unused
            merged[k] = v
            n_copy += 1
    out = dict(pre)
    out["model"] = dict(pre["model"])
    missing = [k for k in merged if k not in out["model"]]
    if missing:
        sys.exit(f"keys not in pretrained checkpoint: {missing[:5]}")
    out["model"].update(merged)
    out["lora_export"] = {"from": ft_path, "rank": mcfg["lora_rank"],
                          "alpha": mcfg["lora_alpha"], "merged": n_lora}
    torch.save(out, out_path)
    print(f"merged {n_lora} LoRA weights, copied {n_copy} tensors -> {out_path}")

    # Everything outside the encoder is the CTC head (w2v_encoder.proj). LoRA A and
    # B only need their shapes: the rebuilt model gets the merged weight and B = 0.
    head_path = out_path[:-3] + ".ctc_head.pt" if out_path.endswith(".pt") else out_path + ".ctc_head.pt"
    head = {k: v for k, v in ft["model"].items() if not k.startswith("w2v_encoder.w2v_model.")}
    torch.save({"cfg": ft["cfg"], "args": ft.get("args"), "task_state": ft.get("task_state"),
                "optimizer_history": ft.get("optimizer_history"),
                "extra_state": ft.get("extra_state"), "head": head, "keys": {k: tuple(v.shape) for k, v in ft["model"].items()},
                "encoder": out_path}, head_path)
    print(f"saved CTC head ({', '.join(head)}) -> {head_path}")


if __name__ == "__main__":
    main()
