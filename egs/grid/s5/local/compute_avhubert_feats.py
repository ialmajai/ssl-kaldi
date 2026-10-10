#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)
# Apache 2.0
"""
AV-HuBERT Visual Feature Extraction for Kaldi
Extracts AV-HuBERT visual hidden-layer features from dlib-tracked lip regions in videos,
using Kaldi I/O (scp/ark).
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import cv2
import dlib
from kaldiio import WriteHelper

# The recipe root's "shared" symlink holds helpers common to all recipes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from kaldi_io_utils import write_utt2dur  # noqa: E402

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
    parser.add_argument(
        "--crop-mode",
        choices=["centroid", "meanface"],
        default="meanface",
        help="ROI crop method. 'meanface' (default): canonical AV-HuBERT similarity-warp "
             "to a mean face, 96->88 centre crop (requires scikit-image). 'centroid': "
             "legacy lip-corner-centred crop, kept to reproduce the archived results; "
             "about 0.5 WER worse but skips the warp (pair it with --crop-scale).",
    )
    parser.add_argument(
        "--mean-face",
        type=str,
        default="av_hubert/avhubert/20words_mean_face.npy",
        help="Mean-face reference landmarks (.npy) for --crop-mode meanface.",
    )
    parser.add_argument(
        "--window-margin",
        type=int,
        default=12,
        help="Temporal smoothing window for the mean-face transform (meanface mode).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=74,
        help="Skip utterances with fewer than this many mouth frames. Lower it for "
             "shorter out-of-domain clips (e.g. Lombard GRID is ~60 frames).",
    )
    parser.add_argument(
        "--mouth-size",
        type=int,
        default=64,
        help="Legacy centroid crop window (px) before resize to 88, when --crop-scale is 0. "
             "The default 64 suits GRID (360x288); fixed sizes do not transfer across "
             "resolutions -- prefer --crop-scale for that.",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=0.0,
        help="Legacy centroid only: if >0, size the crop as round(crop_scale x median "
             "interocular distance) per utterance -- resolution/head-size invariant, "
             "the way icefall normalises. 1.37 reproduces GRID's 64px (IOD ~47).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply offline ROI augmentation (flip/rotate/crop-jitter/blur/noise, "
             "see augment_roi_frames) before AV-HuBERT extraction -- for building "
             "an augmented copy of a training set for multi-condition training, "
             "not for normal feature extraction.",
    )
    parser.add_argument(
        "--augment-seed", type=int, default=0,
        help="Base seed for --augment; combined with each utt-id so the same "
             "seed always reproduces the same augmentation for a given utterance.",
    )
    return parser.parse_args()


def load_avhubert(args: argparse.Namespace, device: torch.device):
    sys.path.insert(0, os.path.join(args.path, "fairseq"))
    sys.path.insert(0, args.path)

    from avhubert.utils import Compose, Normalize
    import avhubert.hubert_pretraining  # noqa: F401  (registers fairseq tasks)
    import avhubert.hubert  # noqa: F401  (registers fairseq tasks)
    import avhubert.hubert_asr  # noqa: F401  (registers fine-tuned model classes)
    from fairseq import checkpoint_utils

    logger.info("Loading AV-HuBERT checkpoint...")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt)
    # Repoint a fine-tuned checkpoint's saved pretrained path (Docker-era
    # /opt/ssl-kaldi) to the same file name next to the checkpoint.
    model_cfg = state["cfg"]["model"]
    w2v = model_cfg["w2v_path"] if "w2v_path" in model_cfg else None
    if w2v and not os.path.exists(w2v):
        state["cfg"]["model"]["w2v_path"] = os.path.join(
            os.path.dirname(os.path.abspath(args.ckpt)), os.path.basename(w2v))
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.ckpt], state=state)
    del state  # free the second copy of the weights (matters for large checkpoints)
    model = models[0]
    if hasattr(model, "encoder") and hasattr(model.encoder, "w2v_model"):
        # fine-tuned seq2seq checkpoint: unwrap the AV-HuBERT encoder
        logger.info("Fine-tuned checkpoint detected; extracting from encoder.w2v_model")
        model = model.encoder.w2v_model
    # Trim to the requested layer BEFORE moving to device: the discarded
    # layers are still CPU tensors at this point (fresh off the checkpoint),
    # so they're never allocated on the GPU at all, rather than being moved
    # there and then discarded. Slicing a ModuleList and reassigning drops
    # the last reference to the removed layers' parameters, so they're
    # garbage-collected here on CPU. This used to run after model.to(device),
    # which briefly moved the full 12-layer model to GPU every time -- for
    # --layer 4 that meant 8 unused layers' worth of parameters (base:
    # ~7M params/layer) touched the GPU and sat in PyTorch's caching
    # allocator pool (visible in nvidia-smi, not torch.cuda.memory_allocated)
    # for the rest of the process, never reclaimed since the allocator
    # doesn't return freed blocks to the driver on its own.
    if args.layer is not None:
        model.encoder.layers = model.encoder.layers[: args.layer]
    model = model.to(device)
    model.eval()
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


def _crop_roi(
    gray: np.ndarray, lm: np.ndarray, h_w: int, h_h: int, mouth_w: int, mouth_h: int
) -> np.ndarray:
    """Crop a full-size mouth ROI around the lip-corner centroid, clamped to the frame."""
    height, width = gray.shape
    cx, cy = _centroid(lm, 48, 54)
    x1 = min(max(0, cx - h_w), max(0, width - mouth_w))
    y1 = min(max(0, cy - h_h), max(0, height - mouth_h))
    roi = gray[y1: y1 + mouth_h, x1: x1 + mouth_w]
    return cv2.resize(roi, (88, 88))


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

    Cache-miss path: single decode pass; landmark detection and ROI crop happen in the
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
            # else: carry last_lm forward so a detection gap doesn't drop frames
            # (which would splice later video earlier in time)

        lm_cache.append(last_lm.copy())

        if np.any(last_lm):
            frames.append(_crop_roi(gray, last_lm, h_w, h_h, mouth_w, mouth_h))

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
        frames.append(_crop_roi(gray, lm, h_w, h_h, mouth_w, mouth_h))

    vidcap.release()
    result = np.stack(frames) if frames else np.empty((0, 88, 88), dtype=np.uint8)
    return result, fps


def centroid_tracking_dyn(video: str, detector, predictor, crop_scale: float,
                          fallback_size: int = 64) -> tuple[np.ndarray, float]:
    """Centroid mouth ROIs with a crop window sized to head scale: crop =
    round(crop_scale x median interocular distance) over the utterance, so the
    mouth-to-ROI ratio is constant across resolutions (no per-dataset tuning).
    Single decode pass (no landmark cache); returns (T, 88, 88) uint8 + fps."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    grays: list = []
    lms: list = []
    last = np.zeros((68, 2), dtype=np.int16)
    while True:
        ok, image = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            last = np.array([[p.x, p.y] for p in predictor(gray, faces[0]).parts()],
                            dtype=np.int16)
        grays.append(gray)
        lms.append(last.copy())
    cap.release()

    iods = []
    for lm in lms:
        if np.any(lm):
            re = lm[36:42].astype(np.float64).mean(0)
            le = lm[42:48].astype(np.float64).mean(0)
            iods.append(float(np.hypot(*(le - re))))
    iod = float(np.median(iods)) if iods else 0.0
    w = max(16, int(round(crop_scale * iod))) if iod > 0 else fallback_size
    hw = w // 2

    frames = [_crop_roi(g, lm, hw, hw, w, w) for g, lm in zip(grays, lms) if np.any(lm)]
    result = np.stack(frames) if frames else np.empty((0, 88, 88), dtype=np.uint8)
    return result, fps


# --- Mean-face alignment (canonical AV-HuBERT preprocessing) --------------------
# Ports av_hubert/avhubert/preparation/align_mouth.py: per-frame dlib landmarks ->
# temporal smoothing -> similarity warp of stable points (eye corners + nose tip)
# onto a mean face in a 256x256 canvas -> 96x96 mouth crop -> 88x88 centre crop.
# The reference functions are reused verbatim so the geometry matches training.
STABLE_PTS = [33, 36, 39, 42, 45]   # nose tip + outer/inner eye corners
STD_SIZE = (256, 256)
_CROP = 96                          # align_mouth crop_height/width
_CENTER = (_CROP - 88) // 2         # 96 -> 88 centre-crop margin


def load_meanface_aligner(args: argparse.Namespace):
    """Import the reference align_mouth functions and load the mean-face landmarks."""
    import importlib.util as il
    am_path = os.path.join(args.path, "avhubert/preparation/align_mouth.py")
    spec = il.spec_from_file_location("align_mouth", am_path)
    am = il.module_from_spec(spec)
    try:
        spec.loader.exec_module(am)
    except ImportError as e:
        raise SystemExit(
            f"--crop-mode meanface needs scikit-image (align_mouth import failed: {e}).\n"
            f"Install it, e.g.:  conda install -c conda-forge scikit-image"
        )
    mean_face = np.load(args.mean_face)
    return am, mean_face


def detect_landmarks_list(video: str, detector, predictor, detect_every: int = 1,
                          fallback_landmarker=None):
    """Per-frame dlib 68-landmarks as a list (None where no face was found), in the
    format align_mouth.landmarks_interpolate expects.

    detect_every: attempt detection every Nth frame while dlib keeps
    succeeding (landmarks_interpolate linearly fills the skipped frames --
    fine while tracking is easy, since a face barely moves frame-to-frame).
    Adaptive, not a fixed stride: any frame where dlib finds no face forces
    every following frame to be attempted again, until dlib succeeds once
    more. This keeps the interpolation gap around an actual trouble spot
    (occlusion, motion blur, pose change) as small as a plain detect_every=1
    pass would give, instead of possibly widening it right when the
    landmarks matter most. Default 1 (attempt every frame, no adaptation)
    keeps every training-feature-extraction caller byte-for-byte unchanged.

    fallback_landmarker: optional callable (BGR frame) -> (68,2) array or None,
    tried only on frames where dlib's detector finds no face. Not used by any
    training-feature-extraction caller (default None keeps them byte-for-byte
    unchanged); wired up by the interactive demo, see kaldi_lipreader.py."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    fps: float = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lms: list = []
    cur = None
    frames_since_attempt = detect_every  # force an attempt on frame 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if frames_since_attempt >= detect_every:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if faces:
                cur = np.array([[p.x, p.y] for p in predictor(gray, faces[0]).parts()],
                               dtype=np.float64)
                frames_since_attempt = 1  # count this frame, so the stride is detect_every
            else:
                # dlib lost the face: leave frames_since_attempt >= detect_every
                # so the very next frame is attempted too, not skipped.
                if fallback_landmarker is not None:
                    fb = fallback_landmarker(image)
                    cur = fb.astype(np.float64) if fb is not None else None
                else:
                    cur = None
        else:
            frames_since_attempt += 1
        lms.append(cur)
    cap.release()
    return lms, fps


def meanface_tracking(video: str, detector, predictor, am, mean_face,
                      window_margin: int = 12, fallback_landmarker=None,
                      detect_every: int = 1) -> tuple[np.ndarray, float]:
    """Mean-face-aligned mouth ROIs. Returns (T, 88, 88) uint8 grayscale frames and
    fps, or an empty array if no face was ever detected.

    The aligned ROI is cached as <video>.mfroi.npz and reused on later calls: it is
    model- and layer-independent, so one cache serves every meanface extraction
    (base, large, any layer) and skips the costly dlib detection + skimage warp.
    A cache entry is only ever written on success, so a fallback_landmarker added
    later still gets a fresh attempt on a video that previously failed outright.

    detect_every: see detect_landmarks_list -- default 1 keeps every
    training-feature-extraction caller unchanged."""
    # Key non-default crops apart so they never reuse the default cache.
    tag = "" if (window_margin, detect_every) == (12, 1) else f".wm{window_margin}de{detect_every}"
    cache = os.path.splitext(video)[0] + tag + ".mfroi.npz"
    if Path(cache).exists():
        try:
            with np.load(cache) as d:
                return d["roi"], float(d["fps"])
        except Exception:
            logger.warning(f"Rebuilding unreadable ROI cache {cache}")

    landmarks, fps = detect_landmarks_list(video, detector, predictor,
                                           detect_every=detect_every,
                                           fallback_landmarker=fallback_landmarker)
    landmarks = am.landmarks_interpolate(landmarks)
    if landmarks is None:            # no face detected on any frame
        return np.empty((0, 88, 88), dtype=np.uint8), fps
    seq = am.crop_patch(video, landmarks, mean_face, STABLE_PTS, STD_SIZE,
                        window_margin=window_margin, start_idx=48, stop_idx=68,
                        crop_height=_CROP, crop_width=_CROP)
    if seq is None:
        return np.empty((0, 88, 88), dtype=np.uint8), fps
    out = np.empty((len(seq), 88, 88), dtype=np.uint8)
    for i, patch in enumerate(seq):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
        out[i] = gray[_CENTER:_CENTER + 88, _CENTER:_CENTER + 88]
    # Write then rename, so a killed job never leaves a truncated cache.
    tmp = f"{cache}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, roi=out, fps=fps)
    os.replace(tmp, cache)
    return out, fps


AVHUBERT_FPS = 25.0


def resample_frames(frames: np.ndarray, fps: float) -> np.ndarray:
    """Pick the frames nearest a 25 fps grid; AV-HuBERT was trained at 25 fps.
    Leaves 25 fps input (all of GRID) untouched; Lombard GRID is about 24 fps."""
    if not fps or abs(fps - AVHUBERT_FPS) < 0.01:
        return frames
    n = int(round(len(frames) * AVHUBERT_FPS / fps))
    idx = np.minimum(np.round(np.arange(n) * fps / AVHUBERT_FPS).astype(int), len(frames) - 1)
    return frames[idx]


@torch.no_grad()
def extract_avhubert(roi_video: torch.Tensor, model, layer: Optional[int]) -> np.ndarray:
    feature_vid, _ = model.extract_finetune(
        source={"video": roi_video, "audio": None},
        padding_mask=None,
        output_layer=layer,
    )
    return feature_vid.squeeze(0).cpu().numpy()


def utt_seed(base_seed: int, utt_id: str) -> int:
    """Deterministic per-utterance seed: same base_seed + utt_id always
    reproduces the same augmentation, independent of job/scp-split order."""
    return (base_seed + zlib.crc32(utt_id.encode())) % (2**31)


def augment_roi_frames(frames: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independently-gated geometric/photometric augmentation for a mouth-ROI
    clip (T, H, W) uint8, at moderate-to-severe severities -- a subset of
    icefall's AVHubertAugment (conformer_ctc2/asr_datamodule.py), which
    trains on noisy/blurred/rotated ROI on the fly and is measurably more
    robust to noise/blur at test time as a result (see NOTES.md's
    "Noise/blur robustness sweep" entry). Kaldi's nnet3/chain training
    consumes fixed pre-extracted feature archives -- there is no per-epoch
    training loop to hook an on-the-fly augmenter into -- so this runs
    offline, once, to build a second (augmented) copy of the training set
    that goes into the mix alongside the clean copy (multi-condition
    training, the standard Kaldi equivalent of speed/volume perturbation).

    Each transform is gated by its own coin flip; a given clip may get any
    combination of these, from none to all five. The same spatial transform
    (flip/rotate/crop) is applied to every frame in the clip for temporal
    consistency, matching icefall's design; only the pixel noise is drawn
    independently per pixel. Severities are wider than icefall's own
    (moderate-to-severe was the explicit ask, not a mild-augmentation
    match): rotation +/-10 deg (icefall: +/-3), blur sigma 0.5-3.0
    (icefall: 0.3-0.6), noise sigma 10-40 in 0-255 pixel space (icefall:
    0.12 in already-normalised float space, roughly sigma~30 on this
    scale -- so comparable at the top of this range, milder at the bottom).
    """
    out = frames
    T, H, W = out.shape

    if rng.random() < 0.5:  # horizontal flip
        out = out[:, :, ::-1]

    if rng.random() < 0.5:  # rotation
        angle = rng.uniform(-10.0, 10.0)
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        out = np.stack([
            cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            for f in out
        ])

    if rng.random() < 0.5:  # crop-resize jitter (scale/shift), up to ~25% zoom
        crop = int(rng.integers(66, 84))
        top = int(rng.integers(0, H - crop + 1))
        left = int(rng.integers(0, W - crop + 1))
        out = np.stack([
            cv2.resize(f[top:top + crop, left:left + crop], (W, H),
                      interpolation=cv2.INTER_LINEAR)
            for f in out
        ])

    if rng.random() < 0.5:  # blur
        sigma = rng.uniform(0.5, 3.0)
        out = np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in out])

    if rng.random() < 0.5:  # noise
        sigma = rng.uniform(10.0, 40.0)
        noise = rng.normal(0.0, sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255)

    return out.astype(np.uint8)


def process_features(
    args: argparse.Namespace,
    model,
    transform,
    detector,
    predictor,
    device: torch.device,
    aligner=None,
    mean_face=None,
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
                    if args.crop_mode == "meanface":
                        roi_frames, fps = meanface_tracking(
                            video_path, detector, predictor, aligner, mean_face,
                            args.window_margin)
                    elif args.crop_scale > 0:
                        roi_frames, fps = centroid_tracking_dyn(
                            video_path, detector, predictor, args.crop_scale,
                            fallback_size=args.mouth_size)
                    else:
                        roi_frames, fps = mouth_tracking(video_path, detector, predictor,
                                                         mouth_w=args.mouth_size,
                                                         mouth_h=args.mouth_size)

                    duration = len(roi_frames) / fps if fps else 0.0
                    roi_frames = resample_frames(roi_frames, fps)
                    if len(roi_frames) < args.min_frames:
                        logger.warning(f"{utt_id}: only {len(roi_frames)} mouth frames, skipping")
                        failed += 1
                        continue

                    if args.augment:
                        rng = np.random.default_rng(utt_seed(args.augment_seed, utt_id))
                        roi_frames = augment_roi_frames(roi_frames, rng)

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
                    utt2dur[utt_id] = duration
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

    if processed == 0:
        logger.error("No utterances were successfully processed")
        sys.exit(1)

    return utt2dur


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

    aligner = mean_face = None
    if args.crop_mode == "meanface":
        logger.info(f"Mean-face alignment enabled (mean face: {args.mean_face})")
        aligner, mean_face = load_meanface_aligner(args)

    utt2dur = process_features(args, model, transform, detector, predictor_model, device,
                               aligner=aligner, mean_face=mean_face)
    write_utt2dur(utt2dur, args.write_utt2dur)
    logger.info("AV-HuBERT feature extraction completed successfully!")


if __name__ == "__main__":
    main()
