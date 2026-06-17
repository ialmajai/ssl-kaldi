#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)
# Apache 2.0
from __future__ import annotations
"""
AV-HuBERT Visual Feature Extraction for Kaldi
Extracts AV-HuBERT visual hidden-layer features from dlib-tracked lip regions in videos,
using Kaldi I/O (scp/ark).
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
import dlib
from kaldiio import WriteHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract AV-HuBERT visual features for Kaldi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input SCP: scp:path/to/video.scp or scp,p:path/to/video.scp",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output: ark:path/to/output.ark or ark:- for stdout",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to AV-HuBERT checkpoint (.pt)")
    parser.add_argument("--path", type=str, required=True, help="Path to cloned AV-HuBERT repo")
    parser.add_argument("-l", "--layer", type=int, default=None, help="Encoder layer to extract")
    parser.add_argument("--write-utt2dur", type=str, default=None, help="Output utt2dur file (ark,t: or ark: prefix)")
    parser.add_argument(
        "--dlib-predictor",
        type=str,
        default="input/shape_predictor_68_face_landmarks.dat",
        help="Path to dlib 68-point face landmark model (.dat)",
    )
    return parser.parse_args()


def load_avhubert(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, os.path.join(args.path, "fairseq"))
    sys.path.insert(0, args.path)

    from avhubert.utils import Compose, Normalize
    import avhubert.hubert_pretraining, avhubert.hubert  # noqa: F401 — registers fairseq tasks
    from fairseq import checkpoint_utils

    logger.info("Loading AV-HuBERT checkpoint...")
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.ckpt])
    model = models[0].to(device)
    model.eval()
    if args.layer is not None:
        model.encoder.layers = model.encoder.layers[: args.layer]
    logger.info(f"Loaded checkpoint: {args.ckpt}  layer={args.layer}")

    transform = Compose([
        Normalize(0.0, 255.0),
        Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])
    return model, transform


def iter_scp(scp_spec: str):
    """Yield (utt_id, video_path) from a scp: or scp,p: spec."""
    colon_idx = scp_spec.index(":")
    scp_path = scp_spec[colon_idx + 1:]
    with open(scp_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            utt_id, video_path = line.split(None, 1)
            yield utt_id, video_path


def _centroid(lm: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """Integer centroid of two landmark points, safe against int16 overflow."""
    return (int(lm[i, 0]) + int(lm[j, 0])) // 2, (int(lm[i, 1]) + int(lm[j, 1])) // 2


def mouth_tracking(
    video: str,
    detector,
    predictor,
    mouth_w: int = 64,
    mouth_h: int = 64,
    detect_every: int = 1,
) -> tuple[np.ndarray, float]:
    """
    Extract mouth ROI frames and FPS from video.

    Cache-miss path: single decode pass — landmark detection and ROI crop happen in the
    same loop, then landmarks are saved. Cache-hit path: single ROI-only decode pass.

    Returns:
        frames: uint8 ndarray of shape (T, 88, 88)
        fps:    frames per second from the container header
    """
    base = os.path.splitext(video)[0]
    landmarks_file = base + ".landmarks.npz"
    h_w, h_h = mouth_w // 2, mouth_h // 2

    if Path(landmarks_file).exists():
        landmarks = np.load(landmarks_file)["landmarks"]
        return _roi_pass(video, landmarks, h_w, h_h, mouth_w, mouth_h)

    # First-run: single-pass detect + crop
    vidcap = cv2.VideoCapture(video)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps: float = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    lm_cache: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    last_lm = np.zeros((68, 2), dtype=np.int16)
    frame_idx = 0

    while True:
        ok, image = vidcap.read()
        if not ok:
            break

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if frame_idx % detect_every == 0:
            faces = detector(gray)
            if faces:
                pts = predictor(gray, faces[0]).parts()
                last_lm = np.array([[p.x, p.y] for p in pts], dtype=np.int16)
            else:
                last_lm = np.zeros((68, 2), dtype=np.int16)

        lm_cache.append(last_lm.copy())

        if np.any(last_lm):
            cx, cy = _centroid(last_lm, 48, 54)
            x1, y1 = max(0, cx - h_w), max(0, cy - h_h)
            roi = gray[y1: y1 + mouth_h, x1: x1 + mouth_w]
            frames.append(cv2.resize(roi, (88, 88)))

        frame_idx += 1

    vidcap.release()
    np.savez_compressed(landmarks_file, landmarks=np.array(lm_cache, dtype=np.int16))

    result = np.stack(frames) if frames else np.empty((0, 88, 88), dtype=np.uint8)
    return result, fps


def _roi_pass(
    video: str,
    landmarks: np.ndarray,
    h_w: int,
    h_h: int,
    mouth_w: int,
    mouth_h: int,
) -> tuple[np.ndarray, float]:
    """ROI-only decode when landmarks are already cached."""
    vidcap = cv2.VideoCapture(video)
    if not vidcap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")

    fps: float = vidcap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[np.ndarray] = []

    for lm in landmarks:
        ok, image = vidcap.read()
        if not ok:
            break
        if not np.any(lm):
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cx, cy = _centroid(lm, 48, 54)
        x1, y1 = max(0, cx - h_w), max(0, cy - h_h)
        roi = gray[y1: y1 + mouth_h, x1: x1 + mouth_w]
        frames.append(cv2.resize(roi, (88, 88)))

    vidcap.release()
    result = np.stack(frames) if frames else np.empty((0, 88, 88), dtype=np.uint8)
    return result, fps


@torch.no_grad()
def extract_avhubert(roi_video: torch.Tensor, model, layer: Optional[int]) -> np.ndarray:
    feature_vid, _ = model.extract_finetune(
        source={"video": roi_video, "audio": None},
        padding_mask=None,
        output_layer=layer,
    )
    return feature_vid.squeeze(0).cpu().numpy()


def process_features(
    args: argparse.Namespace,
    model,
    transform,
    detector,
    predictor,
    device: torch.device,
) -> dict[str, float]:
    logger.info("=" * 70)
    logger.info(f"Input:      {args.input}")
    logger.info(f"Output:     {args.output}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info("=" * 70)

    utt2dur: dict[str, float] = {}
    processed = failed = 0

    try:
        with WriteHelper(args.output) as writer:
            for utt_id, video_path in iter_scp(args.input):
                try:
                    roi_frames, fps = mouth_tracking(video_path, detector, predictor)

                    if len(roi_frames) < 74:
                        logger.warning(f"{utt_id}: only {len(roi_frames)} mouth frames, skipping")
                        failed += 1
                        continue

                    utt2dur[utt_id] = len(roi_frames) / fps

                    roi_frames = transform(roi_frames)
                    tensor = (
                        torch.from_numpy(np.ascontiguousarray(roi_frames))
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device, non_blocking=True)
                    )
                    feats = extract_avhubert(tensor, model, args.layer)
                    writer(utt_id, feats.astype(np.float32))
                    processed += 1

                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 70)
    logger.info(f"Processed: {processed} | Failed: {failed}")
    logger.info("=" * 70)
    return utt2dur


def write_utt2dur(utt2dur: dict[str, float], output_spec: Optional[str]) -> None:
    if not output_spec:
        return
    path = output_spec
    for prefix in ("ark,t:", "ark:"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    logger.info(f"Writing utt2dur: {path}")
    try:
        with open(path, "w") as f:
            for utt_id, dur in sorted(utt2dur.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur)} durations")
    except OSError as e:
        logger.error(f"Failed to write utt2dur: {e}")


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    pred_path = Path(args.dlib_predictor)
    if not pred_path.is_file():
        logger.error(f"dlib predictor not found: {pred_path}")
        sys.exit(1)

    logger.info(f"Loading dlib models from {pred_path}")
    logger.info(f"dlib CUDA: {dlib.DLIB_USE_CUDA} | devices: {dlib.cuda.get_num_devices()}")
    detector = dlib.get_frontal_face_detector()
    predictor_model = dlib.shape_predictor(str(pred_path))

    model, transform = load_avhubert(args, device)

    utt2dur = process_features(args, model, transform, detector, predictor_model, device)
    write_utt2dur(utt2dur, args.write_utt2dur)
    logger.info("AV-HuBERT feature extraction completed successfully!")


if __name__ == "__main__":
    main()
