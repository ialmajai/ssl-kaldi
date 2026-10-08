#!/usr/bin/env python3
# Copyright   2026  (author: Ibrahim Almajai)
# Apache 2.0
"""
Build the <word-feats> input steps/conf/{train,apply}_calibration.sh need:
one line per word-id, "word word_id filter length unigram_logprob".

- filter: 1 for real vocabulary words (scored in calibration training and
  kept in calibrated output), 0 for <eps>/#N disambig/<s>/</s> (Kaldi's
  standard non-content words.txt entries -- excluded, matching
  prepare_calibration_data.py's own word_filter use).
- length: phone count from the lexicon's align_lexicon.txt (word word
  phone1 phone2 ...); 1 for anything missing from the lexicon (only the
  filtered-out non-content entries), to keep log(length) finite.
- unigram_logprob: log10 unigram probability from a training-text word
  count, add-1 smoothed over the words.txt vocabulary so unseen words
  never hit log(0).
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

NON_CONTENT_PREFIXES = ("#",)
NON_CONTENT_WORDS = {"<eps>", "<s>", "</s>"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("words_txt")
    parser.add_argument("align_lexicon_txt")
    parser.add_argument("train_text")
    parser.add_argument("out_word_feats")
    args = parser.parse_args()

    word_ids: dict[str, int] = {}
    for line in Path(args.words_txt).read_text().splitlines():
        w, i = line.split()
        word_ids[w] = int(i)

    phone_count: dict[str, int] = {}
    for line in Path(args.align_lexicon_txt).read_text().splitlines():
        parts = line.split()
        w = parts[0]
        phone_count[w] = len(parts) - 2  # "word word phone1 phone2 ..."

    counts = Counter()
    for line in Path(args.train_text).read_text().splitlines():
        for w in line.split()[1:]:  # skip utt-id
            counts[w] += 1
    vocab_size = len(word_ids)
    total = sum(counts.values()) + vocab_size  # add-1 smoothing

    with open(args.out_word_feats, "w") as f:
        for w, wid in sorted(word_ids.items(), key=lambda kv: kv[1]):
            is_content = w not in NON_CONTENT_WORDS and not w.startswith(NON_CONTENT_PREFIXES)
            filt = 1 if is_content else 0
            length = phone_count.get(w, 1)
            logprob = math.log10((counts[w] + 1) / total)
            f.write(f"{w} {wid} {filt} {length} {logprob:.6f}\n")


if __name__ == "__main__":
    main()
