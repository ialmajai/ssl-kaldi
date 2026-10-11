#!/usr/bin/env python3
"""Dump 96x96 mean-face mouth ROIs for AV-HuBERT fine-tuning.

Same warp as compute_avhubert_feats.py --crop-mode meanface (align_mouth.crop_patch
onto the mean face), but keeps the full 96x96 crop so training can take random
88x88 crops; its centre 88x88 is what feature extraction feeds the encoder.
Landmarks come from the <video>.landmarks.npz cache, or from fresh dlib
detection (as extraction does) when that cache is missing. Frames are written
losslessly (FFV1 .avi) so training sees the same pixels extraction does.
  dump_roi_meanface.py video.scp out_dir [--nj N]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import compute_avhubert_feats as cav  # noqa: E402

AM = MEAN_FACE = DETECTOR = PREDICTOR = None


def init(path, mean_face, predictor):
    global AM, MEAN_FACE, DETECTOR, PREDICTOR
    import dlib
    AM, MEAN_FACE = cav.load_meanface_aligner(
        argparse.Namespace(path=path, mean_face=mean_face))
    DETECTOR, PREDICTOR = dlib.get_frontal_face_detector(), dlib.shape_predictor(predictor)


def roi96(video):
    cache = os.path.splitext(video)[0] + ".landmarks.npz"
    if os.path.exists(cache):
        lms = [x.astype(np.float64) for x in np.load(cache)["landmarks"]]
    else:
        lms, _ = cav.detect_landmarks_list(video, DETECTOR, PREDICTOR)
    landmarks = AM.landmarks_interpolate(lms)
    if landmarks is None:
        return None
    seq = AM.crop_patch(video, landmarks, MEAN_FACE, cav.STABLE_PTS, cav.STD_SIZE,
                        window_margin=12, start_idx=48, stop_idx=68,
                        crop_height=cav._CROP, crop_width=cav._CROP)
    if seq is None:
        return None
    return np.stack([cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) if p.ndim == 3 else p for p in seq])


def write(path, frames, fps=25.0):
    h, w = frames.shape[1:]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h), isColor=False)
    for f in frames:
        vw.write(f)
    vw.release()


def job(item):
    utt, video, out = item
    dst = os.path.join(out, utt + ".avi")
    if os.path.exists(dst):
        return utt, "skip"
    try:
        frames = roi96(video)
    except Exception as e:  # noqa: BLE001
        return utt, f"error {e}"
    if frames is None or len(frames) == 0:
        return utt, "noface"
    tmp = dst + f".tmp{os.getpid()}.avi"
    write(tmp, frames)
    os.replace(tmp, dst)   # atomic: a concurrent dumper never sees a partial file
    return utt, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_scp")
    ap.add_argument("out_dir")
    ap.add_argument("--nj", type=int, default=8)
    ap.add_argument("--path", default="av_hubert")
    ap.add_argument("--mean-face", default="av_hubert/avhubert/20words_mean_face.npy")
    ap.add_argument("--dlib-predictor", default="input/shape_predictor_68_face_landmarks.dat")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    items = [(l.split()[0], l.split()[1], a.out_dir) for l in open(a.video_scp)]
    counts = {}
    with Pool(a.nj, initializer=init, initargs=(a.path, a.mean_face, a.dlib_predictor)) as pool:
        for i, (utt, st) in enumerate(pool.imap_unordered(job, items, chunksize=16)):
            k = st.split()[0]
            counts[k] = counts.get(k, 0) + 1
            if k == "error":
                print(utt, st, file=sys.stderr)
            if (i + 1) % 2000 == 0:
                print(f"{i + 1}/{len(items)} {counts}", flush=True)
    print(f"done {counts}", flush=True)


if __name__ == "__main__":
    main()
