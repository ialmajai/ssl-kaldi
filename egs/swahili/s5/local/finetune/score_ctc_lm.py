#!/usr/bin/env python3
# Copyright  2026  Ibrahim Almajai
# Apache 2.0
"""Decode a CTC-fine-tuned model with a KenLM word LM and score WER.

Companion to score_ctc.py (greedy, no LM): same model and data handling, but
the character posteriors go through pyctcdecode beam search with the recipe's
own ARPA LM and vocabulary. Logits are computed once and cached in memory, so
the alpha/beta grid costs only CPU decode time; the grid is tuned on a subset
and the best pair scored on the full set (the same tune-on-dev convention as
the chain's wer_* LMWT grid).

Usage:
    python local/finetune/score_ctc_lm.py --ctc-dir exp_ft/wavlmbp_lora32ffn/ctc \\
        --data-dir data/dev_clean_2 --arpa /tmp/lm_tgsmall.arpa
"""
import argparse
import json
import os

import numpy as np
import torch
from kaldiio import ReadHelper
from pyctcdecode import build_ctcdecoder
from transformers import AutoFeatureExtractor, AutoModelForCTC

EXPECTED_SAMPLE_RATE = 16000
BLANK = 0
WORD_DELIM = "|"


def parse_args():
    p = argparse.ArgumentParser(
        description="Beam-search CTC decoding with a word LM, WER scoring",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ctc-dir", required=True)
    p.add_argument("--data-dir", default="data/dev_clean_2")
    p.add_argument("--arpa", required=True, help="decompressed ARPA (or kenlm .bin)")
    p.add_argument("--unigrams", default="data/local/lm/librispeech-vocab.txt")
    p.add_argument("--vocab", default=None,
                   help="chars.json from the fine-tune (defaults to the parent "
                        "of --ctc-dir)")
    p.add_argument("--batch-utts", type=int, default=2)
    p.add_argument("--beam-width", type=int, default=100)
    p.add_argument("--tune-utts", type=int, default=300,
                   help="tune alpha/beta on this many utterances, then score "
                        "the full set with the best pair")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.6, 1.0, 1.5])
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.5, 3.0])
    return p.parse_args()


def levenshtein(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer_of(decoder, logp, refs, idx, alpha, beta, beam):
    decoder.reset_params(alpha=alpha, beta=beta)
    errs = length = 0
    for i in idx:
        hyp = decoder.decode(logp[i], beam_width=beam).split()
        errs += levenshtein(refs[i], hyp)
        length += len(refs[i])
    return 100.0 * errs / max(length, 1), errs, length


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab_path = args.vocab or os.path.join(
        os.path.dirname(args.ctc_dir.rstrip("/")), "chars.json")
    with open(vocab_path) as f:
        vocab = json.load(f)
    # pyctcdecode: labels indexed by class id; blank is "", WORD_DELIM is space.
    labels = [""] * (len(vocab) + 1)
    for c, i in vocab.items():
        labels[i] = " " if c == WORD_DELIM else c

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
    refs = [text[u].split() for u in utts]

    print(f"computing logits for {len(utts)} utterances")
    order = sorted(range(len(waves)), key=lambda i: len(waves[i]))
    logp = [None] * len(utts)
    with torch.no_grad():
        for s in range(0, len(order), args.batch_utts):
            idx = order[s:s + args.batch_utts]
            batch = extractor([waves[i] for i in idx],
                              sampling_rate=EXPECTED_SAMPLE_RATE,
                              return_tensors="pt", padding=True)
            lens = [len(waves[i]) for i in idx]
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device == "cuda"):
                logits = model(**batch).logits
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for row, i in enumerate(idx):
                # trim padding frames: conv stride is 320 samples per frame
                nframes = model._get_feat_extract_output_lengths(
                    torch.tensor(lens[row])).item()
                logp[i] = lp[row, :nframes]

    with open(args.unigrams) as f:
        unigrams = [w.strip() for w in f if w.strip()]
    print(f"building decoder: {args.arpa}, {len(unigrams)} unigrams")
    decoder = build_ctcdecoder(labels, kenlm_model_path=args.arpa,
                               unigrams=unigrams)

    tune_idx = list(range(min(args.tune_utts, len(utts))))
    best = (None, None, float("inf"))
    for a in args.alphas:
        for b in args.betas:
            w, _, _ = wer_of(decoder, logp, refs, tune_idx, a, b,
                             args.beam_width)
            print(f"  tune alpha={a} beta={b}: WER {w:.2f}% "
                  f"({len(tune_idx)} utts)")
            if w < best[2]:
                best = (a, b, w)
    a, b, _ = best
    print(f"best on tune set: alpha={a} beta={b}")

    w, errs, length = wer_of(decoder, logp, refs, range(len(utts)), a, b,
                             args.beam_width)
    print(f"{args.data_dir}: WER = {w:.2f}%  ({errs} errors / {length} words, "
          f"{len(utts)} utterances, beam {args.beam_width}, "
          f"LM {os.path.basename(args.arpa)}, alpha={a} beta={b})")


if __name__ == "__main__":
    main()
