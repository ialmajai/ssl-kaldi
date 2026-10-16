#!/usr/bin/env python3
"""Greedy CTC decode of an avh_lora fine-tune on a set of 96x96 ROI videos.

No LM, no grammar: argmax per frame, collapse repeats, drop blanks, '|' -> space.
Writes '<utt> <words>' lines for compute-wer. Centre 88x88 crop (image_aug off).
  decode_ctc.py CKPT ROI_DIR DATA_DIR OUT_HYP [--manifest-dir D]
  decode_ctc.py MERGED.pt ROI_DIR DATA_DIR OUT_HYP --head MERGED.ctc_head.pt ...
With --head, CKPT is the merged encoder from export_lora_encoder.py and the
fine-tuned model is rebuilt from it and the saved CTC head.
"""
import argparse
import os
import shutil
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("roi_dir")
    ap.add_argument("data_dir", help="Kaldi dir with text and utt2num_frames")
    ap.add_argument("out_hyp")
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--label-dict", required=True, help="dict.ltr.txt from training")
    ap.add_argument("--head", help="CTC head from export_lora_encoder.py; CKPT is then the merged encoder")
    a = ap.parse_args()
    if a.head:
        a.ckpt = rebuild_checkpoint(a.ckpt, a.head, a.manifest_dir)

    os.makedirs(a.manifest_dir, exist_ok=True)
    text = dict(l.rstrip("\n").split(None, 1) for l in open(f"{a.data_dir}/text"))
    nfr = dict(l.split() for l in open(f"{a.data_dir}/utt2num_frames"))
    utts = sorted(u for u in text if u in nfr and os.path.exists(f"{a.roi_dir}/{u}.avi"))
    with open(f"{a.manifest_dir}/test.tsv", "w") as t, open(f"{a.manifest_dir}/test.ltr", "w") as l:
        print("/", file=t)
        for u in utts:
            p = os.path.abspath(f"{a.roi_dir}/{u}.avi")
            n = int(nfr[u])
            print(f"{u}\t{p}\t{p}\t{n}\t{n * 4}", file=t)
            print(" | ".join(" ".join(w) for w in text[u].split()) + " |", file=l)
    shutil.copy(a.label_dict, f"{a.manifest_dir}/dict.ltr.txt")

    from fairseq import checkpoint_utils, utils
    plugin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "avh_lora")
    utils.import_user_module(argparse.Namespace(user_dir=plugin))  # registers the task and model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [a.ckpt], arg_overrides={"data": a.manifest_dir, "label_dir": a.manifest_dir})
    model = models[0].eval().cuda().half()
    task.load_dataset("test")
    ds = task.dataset("test")
    itr = task.get_batch_iterator(dataset=ds, max_tokens=2000, ignore_invalid_inputs=False,
                                  num_workers=4).next_epoch_itr(shuffle=False)
    d = task.target_dictionary
    blank = 0
    out = {}
    with torch.no_grad():
        for sample in itr:
            sample = utils.move_to_cuda(sample)
            sample["net_input"]["source"]["video"] = sample["net_input"]["source"]["video"].half()
            net = model(**sample["net_input"])
            lp = model.get_normalized_probs(net, log_probs=True).transpose(0, 1)  # B x T x V
            pad = net["padding_mask"]
            for i, uid in enumerate(sample["utt_id"]):
                ids = lp[i].argmax(-1)
                if pad is not None:
                    ids = ids[~pad[i]]
                ids = torch.unique_consecutive(ids).tolist()
                chars = [d[x] for x in ids if x != blank and x not in (d.pad(), d.eos(), d.bos())]
                out[uid] = " ".join("".join(chars).replace("|", " ").split())
    with open(a.out_hyp, "w") as f:
        for u in utts:
            print(u, out.get(u, ""), file=f)
    print(f"decoded {len(out)} of {len(utts)}", file=sys.stderr)


def rebuild_checkpoint(enc_path, head_path, work_dir):
    """Write a fairseq checkpoint equivalent to the fine-tune's from the merged
    encoder and the saved head. Each LoRA-wrapped weight gets the merged weight as
    its base and B = 0, so the adapter adds nothing on top of it."""
    h = torch.load(head_path, map_location="cpu")
    enc = torch.load(enc_path, map_location="cpu")["model"]
    pre = "w2v_encoder.w2v_model."
    state = {}
    for k, shape in h["keys"].items():
        if not k.startswith(pre):
            v = h["head"][k]
        elif ".parametrizations.weight.0." in k:          # LoRA A or B
            v = torch.zeros(shape)
        elif k.endswith(".parametrizations.weight.original"):
            v = enc[k[len(pre):-len(".parametrizations.weight.original")] + ".weight"]
        else:
            v = enc[k[len(pre):]]
        if tuple(v.shape) != tuple(shape):
            sys.exit(f"shape mismatch for {k}: {tuple(v.shape)} vs {shape}")
        state[k] = v
    os.makedirs(work_dir, exist_ok=True)
    out = os.path.join(work_dir, "rebuilt_checkpoint.pt")
    torch.save({"cfg": h["cfg"], "args": h["args"], "task_state": h["task_state"],
                "optimizer_history": h["optimizer_history"], "extra_state": h["extra_state"],
                "model": state}, out)
    return out


if __name__ == "__main__":
    main()
