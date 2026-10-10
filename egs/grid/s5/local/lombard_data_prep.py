#!/usr/bin/env python3
# Copyright 2026  Ibrahim Almajai
"""Build a Kaldi data dir for Lombard GRID (one speaking condition) from the
corpus's front-view videos, <spk>_<cond>_<code>.mov. Each six-letter code spells
its GRID sentence (command colour preposition letter digit adverb), so the
transcript is decoded from the file name.

Drops the utterances in --exclude: 69 clips that the reported Lombard results
leave out, rejected by the icefall demo reader (face detection failed or the clip
fell outside 2-15 s). Pass --exclude "" to keep every clip. Feature extraction then
skips 3 more clips with no trackable mouth, so 2,577 utterances are scored.

    local/lombard_data_prep.py --video-dir lombardgrid/front --cond l --out data/lomb_l
"""
import argparse
import os
import re

COMMAND = {"b": "bin", "l": "lay", "p": "place", "s": "set"}
COLOUR = {"b": "blue", "g": "green", "r": "red", "w": "white"}
PREPOSITION = {"a": "at", "b": "by", "i": "in", "w": "with"}
DIGIT = {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
         "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
ADVERB = {"a": "again", "n": "now", "p": "please", "s": "soon"}
NAME = re.compile(r"^(s\d+)_([a-z])_([a-z0-9]{6})\.mov$")


def decode(code):
    """Return the GRID sentence for a six-letter code, or None if malformed."""
    c, o, p, letter, d, a = code
    if not ("a" <= letter <= "z" and letter != "w"):
        return None
    try:
        return " ".join([COMMAND[c], COLOUR[o], PREPOSITION[p], letter, DIGIT[d], ADVERB[a]])
    except KeyError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="dir of <spk>_<cond>_<code>.mov (lombardgrid/front)")
    ap.add_argument("--cond", default="l", help="condition to keep: l (lombard) or p (plain)")
    ap.add_argument("--exclude", default="local/lombard_exclude.txt",
                    help="file of utt ids to drop; empty string keeps all")
    ap.add_argument("--out", required=True, help="output data dir")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    exclude = set(open(args.exclude).read().split()) if args.exclude else set()

    rows, bad = [], []
    for name in os.listdir(args.video_dir):
        m = NAME.match(name)
        if not m or m.group(2) != args.cond:
            continue
        spk, cond, code = m.groups()
        text = decode(code)
        if text is None:
            bad.append(name)
            continue
        utt = f"{spk}-{cond}-{code}"  # dashes keep utt order == speaker order
        if utt in exclude:
            continue
        rows.append((utt, spk, os.path.abspath(os.path.join(args.video_dir, name)), text))

    rows.sort()
    spk2utt = {}
    with open(f"{args.out}/video.scp", "w") as fv, \
         open(f"{args.out}/text", "w") as ft, \
         open(f"{args.out}/utt2spk", "w") as fu:
        for utt, spk, vid, text in rows:
            print(f"{utt} {vid}", file=fv)
            print(f"{utt} {text}", file=ft)
            print(f"{utt} {spk}", file=fu)
            spk2utt.setdefault(spk, []).append(utt)
    with open(f"{args.out}/spk2utt", "w") as fs:
        for spk in sorted(spk2utt):
            print(spk, " ".join(spk2utt[spk]), file=fs)
    if bad:
        print(f"lombard_data_prep: skipped {len(bad)} undecodable names, e.g. {bad[0]}")
    print(f"lombard_data_prep: wrote {len(rows)} utts (cond={args.cond}) to {args.out}")


if __name__ == "__main__":
    main()
