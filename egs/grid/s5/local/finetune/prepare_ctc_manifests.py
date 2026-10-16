#!/usr/bin/env python3
"""Manifests for AV-HuBERT CTC fine-tuning on GRID (letters, '|' between words).

Validation holds out whole training speakers, so checkpoint choice tracks
unseen-speaker performance rather than fit to the training speakers.
tsv rows: id, video, audio, nframes, audio samples. Without --audio-dir the audio
column repeats the video path (video-only fine-tuning) and the sample count is
nframes*4; with it, the audio is <audio-dir>/<spk>/<clip>.wav (16 kHz, as
AV-HuBERT's loader requires) and clips without a wav are dropped.
  prepare_ctc_manifests.py --data data/train_l4_raw --roi-dir R --out O [--audio-dir A]
"""
import argparse
import os
import random
import wave
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train_l4_raw")
    ap.add_argument("--roi-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid-spk", default="s7,s30")
    ap.add_argument("--valid-per-spk", type=int, default=400)
    ap.add_argument("--audio-dir", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    text = {l.split(None, 1)[0]: l.split(None, 1)[1].strip() for l in open(f"{a.data}/text")}
    spk = dict(l.split() for l in open(f"{a.data}/utt2spk"))
    nfr = {u: int(n) for u, n in (l.split() for l in open(f"{a.data}/utt2num_frames"))}
    held = set(a.valid_spk.split(","))
    def wav(u):
        return os.path.join(a.audio_dir, spk[u], u.split("-", 1)[1] + ".wav")
    utts = sorted(u for u in text if u in nfr
                  and os.path.exists(os.path.join(a.roi_dir, u + ".avi"))
                  and (not a.audio_dir or os.path.exists(wav(u))))
    random.seed(0)
    train = [u for u in utts if spk[u] not in held]
    valid = []
    for s in sorted(held):
        su = [u for u in utts if spk[u] == s]
        random.shuffle(su)
        valid += sorted(su[:a.valid_per_spk])
    counts = Counter()
    for name, subset in [("train", train), ("valid", valid)]:
        with open(f"{a.out}/{name}.tsv", "w") as t, open(f"{a.out}/{name}.ltr", "w") as l, \
             open(f"{a.out}/{name}.wrd", "w") as w:
            print("/", file=t)
            for u in subset:
                p = os.path.abspath(os.path.join(a.roi_dir, u + ".avi"))
                if a.audio_dir:
                    ap_ = os.path.abspath(wav(u))
                    with wave.open(ap_) as f:
                        assert f.getframerate() == 16000, ap_
                        n = f.getnframes()
                else:
                    ap_, n = p, nfr[u] * 4
                print(f"{u}\t{p}\t{ap_}\t{nfr[u]}\t{n}", file=t)
                letters = " | ".join(" ".join(word) for word in text[u].split()) + " |"
                print(letters, file=l)
                print(text[u], file=w)
                if name == "train":
                    counts.update(letters.split())
    with open(f"{a.out}/dict.ltr.txt", "w") as f:
        for tok, c in counts.most_common():
            print(tok, c, file=f)
    print(f"train={len(train)} valid={len(valid)} (held-out {sorted(held)}) letters={len(counts)}")


if __name__ == "__main__":
    main()
