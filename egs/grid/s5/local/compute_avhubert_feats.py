#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0
"""
AV-HuBERT Visual Feature Extraction for Kaldi
--------------------------------------------
Extracts AV-HuBERT visual hidden-layer features from dlib tracked lip-regions in videos,
using Kaldi I/O (scp/ark).

"""

import sys
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import cv2
import dlib
from kaldiio import ReadHelper, WriteHelper

# ------------------------------------------------------------------------- #
# Logging
# ------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- #
# Argument parsing
# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Extract AV-HuBERT visual features for Kaldi",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "input",
    type=str,
    help="Input: scp:path/to/video.scp or ark:- for stdin "
         "(values should be video paths",
)
parser.add_argument(
    "output",
    type=str,
    help="Output: ark:path/to/output.ark or ark:- for stdout",
)

parser.add_argument(
    "--ckpt",
    type=str,
    required=True,
    help="Path to AV-HuBERT checkpoint (.pt)",
)

parser.add_argument(
    "--path",
    type=str,
    required=True,
    help="Path to cloned AV-HuBERT",
)
parser.add_argument(
    "-l",
    "--layer",
    type=int,
    default=None,
    help="Encoder layer to extract ",
)

parser.add_argument(
    "--write-utt2dur",
    type=str,
    default=None,
    help="Output duration file: ark,t:path/to/utt2dur or ark:path",
)
parser.add_argument(
    "--dlib-predictor",
    type=str,
    default="input/shape_predictor_68_face_landmarks.dat",
    help="Path to dlib 68-point face landmark model (.dat).",
)

args = parser.parse_args()


avhubert_path = args.path
sys.path.insert(0, avhubert_path + '/fairseq')
sys.path.insert(0, avhubert_path )
from argparse import Namespace
import fairseq
from avhubert.utils import Compose, CenterCrop, Normalize
import avhubert.hubert_pretraining, avhubert.hubert
from fairseq import checkpoint_utils, options, tasks, utils


# ------------------------------------------------------------------------- #
# Device
# ------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info(dlib.DLIB_USE_CUDA)
logger.info(dlib.cuda.get_num_devices())

# ------------------------------------------------------------------------- #
# dlib: face detector + landmark predictor
# ------------------------------------------------------------------------- #
pred_path = Path(args.dlib_predictor)
if not pred_path.is_file():
    logger.error(f"dlib predictor not found: {pred_path}")
    sys.exit(1)

logger.info(f"Loading dlib face detector and landmark predictor from {pred_path}")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(pred_path))

logger.info("Loading AV-HuBERT model…")
# load AV-HuBERT
logger.info("Loading AV-HuBERT checkpoint...")
models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([args.ckpt])
model = models[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model.eval()
# Remove unused layers 
model.encoder.layers = model.encoder.layers[:args.layer]
logger.info(f"Loaded checkpoint from {args.ckpt}")    

transform = Compose([
    Normalize(0.0, 255.0),
    Normalize(task.cfg.image_mean, task.cfg.image_std),
])

# ------------------------------------------------------------------------- #
# AV-HuBERT feature extraction (visual)
# ------------------------------------------------------------------------- #
@torch.no_grad()
def extract_avhubert(roi_video, model, layer):
    # Extract features
    with torch.no_grad():
        feature_vid, _ = model.extract_finetune(source={'video': roi_video,
                                        'audio': None}, padding_mask=None, output_layer=args.layer)
    feats = feature_vid.squeeze(0)
    return feats.cpu().numpy()

def calculate_duration(num_frames: int, fps: float) -> float:
    return float(num_frames) / float(fps)

def iter_scp_lines(scp_spec):
    if scp_spec.startswith("scp:"):
        scp_path = scp_spec[len("scp:"):]
    else:
        scp_path = scp_spec
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            utt_id, video_path = line.split(None, 1)
            yield utt_id, video_path
            
# ------------------------------------------------------------------------- #
# Face tracking and ROI extraction
# ------------------------------------------------------------------------- #
def mouth_tracking(video, mouth_w=64, mouth_h=64, detect_every=1):
    """Extract mouth region of interest (ROI) from video frames using dlib face detection.

    Args:
        video (str): Path to the input video file.
        mouth_w (int, optional): Width of the mouth ROI. Defaults to 64.
        mouth_h (int, optional): Height of the mouth ROI. Defaults to 64.
        detect_every (int, optional): Detect face every N frames. Defaults to 3.

    Returns:
        np.ndarray: Stack of mouth ROI frames resized to 88x88 pixels.
    """
   
    base, _ = os.path.splitext(video)
    landmarks_file = base + ".landmarks.npz"

    if Path(landmarks_file).exists():
        landmarks = np.load(landmarks_file)["landmarks"]
    else:
        vidcap = cv2.VideoCapture(video)
        landmarks = []
        frame_idx = 0

        while vidcap.isOpened():
            success, image = vidcap.read()
            if not success:
                break

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if frame_idx % detect_every == 0:
                faces = detector(gray)
                if len(faces) == 0:
                    lm = np.zeros((68, 2), dtype=np.int16)
                    landmarks.append(lm)
                else:
                    shape = predictor(gray, faces[0])
                    lm = np.array([[p.x, p.y] for p in shape.parts()])
                    landmarks.append(lm)
            else:
                landmarks.append(landmarks[-1])

            frame_idx += 1

        vidcap.release()
                
        np.savez_compressed(landmarks_file, landmarks=np.array(landmarks, dtype=np.int16))
        
    # ---------------------------
    # ROI extraction from landmarks
    # ---------------------------
    frames = []
    h_width = mouth_w // 2
    h_height = mouth_h // 2

    vidcap = cv2.VideoCapture(video)
    for lm in landmarks:
        success, image = vidcap.read()
        if not success or not np.any(lm):
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cx = (lm[48, 0] + lm[54, 0]) // 2
        cy = (lm[48, 1] + lm[54, 1]) // 2

        x1, y1 = max(0, cx - h_width), max(0, cy - h_height)
        x2, y2 = cx + h_width, cy + h_height

        roi = gray[y1:y2, x1:x2]
        frames.append(cv2.resize(roi, (88, 88)))

    vidcap.release()
    return np.stack(frames)

# ------------------------------------------------------------------------- #
# Main Kaldi processing loop
# ------------------------------------------------------------------------- #
def process_features() -> dict:
    logger.info("=" * 70)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Checkpoint: {args.ckpt}")
    logger.info("=" * 70)

    utt2dur_data = {}
    processed_count = 0
    failed_count = 0

    try:
        with WriteHelper(args.output) as writer:
            for utt_id, video_path in iter_scp_lines(args.input.split(':')[1]):
                try:
                    roi_frames = mouth_tracking(video_path) 
                    if len(roi_frames) < 74: 
                        logger.warning(f"Mouth tracking failed for {utt_id}, skipping...")
                        failed_count += 1
                        continue
                        
                    num_frames = len(roi_frames)
                    
                    duration = calculate_duration(num_frames, 25)
                    utt2dur_data[utt_id] = duration
                    roi_frames = transform(roi_frames)
                      
                    roi_frames = torch.FloatTensor(roi_frames).unsqueeze(dim=0).unsqueeze(dim=0).to(device)                  
                    feats = extract_avhubert(roi_frames, model, args.layer)                     

                    writer(utt_id, feats.astype(np.float32))

                    processed_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")             
                    failed_count += 1
                    continue
                    
        logger.info("=" * 70)
        logger.info(f"Successfully processed: {processed_count} utterances")
        if failed_count > 0:
            logger.warning(f"Failed: {failed_count} utterances")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)

    return utt2dur_data

# ------------------------------------------------------------------------- #
# utt2dur writer
# ------------------------------------------------------------------------- #
def write_utt2dur(utt2dur_data: dict) -> None:
    if not args.write_utt2dur:
        return

    output_path = args.write_utt2dur
    if output_path.startswith("ark,t:"):
        output_path = output_path[6:]
    elif output_path.startswith("ark:"):
        output_path = output_path[4:]

    logger.info(f"Writing duration file: {output_path}")
    try:
        with open(output_path, "w") as f:
            for utt_id, dur in sorted(utt2dur_data.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur_data)} durations to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write utt2dur: {e}")

# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    utt2dur = process_features()
    write_utt2dur(utt2dur)
    logger.info("AV-HuBERT feature extraction completed successfully!")
    

# %%
