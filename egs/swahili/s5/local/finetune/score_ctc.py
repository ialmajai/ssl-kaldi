#!/usr/bin/env python3
# Copyright  2026  Ibrahim Almajai
# Apache 2.0
"""Decode a CTC-fine-tuned model directly and score WER.

Port of egs/timit/s5/local/finetune/score_ctc.py to word-level data: greedy
CTC over character logits, hypotheses re-joined into words at the `|`
delimiter, scored as word error rate against the Kaldi `text`. No language
model, no Kaldi decoding; this is the direct-decoding arm of the
frozen-vs-fine-tuned comparison.

Usage:
    python local/finetune/score_ctc.py --ctc-dir exp_ft/mms300m_lora32/ctc \\
        --data-dir data/dev_clean_2
"""
import argparse
import json
import os

import numpy as np
import torch
from kaldiio import ReadHelper
from transformers import AutoFeatureExtractor, AutoModelForCTC

EXPECTED_SAMPLE_RATE = 16000
BLANK = 0
WORD_DELIM = "|"


def parse_args():
    p = argparse.ArgumentParser(
        description="Greedy CTC decoding and WER scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ctc-dir", required=True,
                   help="directory holding the full *ForCTC model (the 'ctc' "
                        "subdirectory written by finetune_ctc.py)")
    p.add_argument("--data-dir", default="data/dev_clean_2")
    p.add_argument("--vocab", default=None,
                   help="chars.json from the fine-tune (defaults to the "
                        "parent of --ctc-dir)")
    p.add_argument("--batch-utts", type=int, default=2)
    return p.parse_args()


def levenshtein(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def chars_to_words(chars):
    return [w for w in "".join(chars).split(WORD_DELIM) if w]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = args.vocab or os.path.join(
        os.path.dirname(args.ctc_dir.rstrip("/")), "chars.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    inv = {i: c for c, i in vocab.items()}

    extractor = AutoFeatureExtractor.from_pretrained(args.ctc_dir)
    model = AutoModelForCTC.from_pretrained(args.ctc_dir).to(device).eval()

    with open(os.path.join(args.data_dir, "text")) as f:
        text = dict(line.strip().split(None, 1) for line in f if line.strip())

    utts, waves = [], []
    with ReadHelper(f"scp:{os.path.join(args.data_dir, 'wav.scp')}") as reader:
        for utt, (rate, wav) in reader:
            if rate != EXPECTED_SAMPLE_RATE:
                raise ValueError(f"{utt}: {rate} != {EXPECTED_SAMPLE_RATE}")
            if np.issubdtype(wav.dtype, np.integer):
                wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
            utts.append(utt)
            waves.append(wav.astype(np.float32))

    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    errs = length = 0
    with torch.no_grad():
        for s in range(0, len(order), args.batch_utts):
            idx = order[s:s + args.batch_utts]
            batch = extractor([waves[i] for i in idx],
                              sampling_rate=EXPECTED_SAMPLE_RATE,
                              return_tensors="pt", padding=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**batch).logits
            else:
                logits = model(**batch).logits
            pred = logits.argmax(-1).cpu()
            for row, i in enumerate(idx):
                ids, prev = [], None
                for t in pred[row].tolist():
                    if t != prev and t != BLANK:
                        ids.append(t)
                    prev = t
                hyp = chars_to_words([inv[t] for t in ids if t in inv])
                ref = text[utts[i]].split()
                errs += levenshtein(ref, hyp)
                length += len(ref)

    print(f"{args.data_dir}: WER = {100.0 * errs / max(length, 1):.2f}%  "
          f"({errs} errors / {length} words, {len(utts)} utterances, "
          f"greedy CTC, no LM)")


if __name__ == "__main__":
    main()
