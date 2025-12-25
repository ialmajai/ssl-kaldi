#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0
"""
HuBERT Feature Extraction for Kaldi
-----------------------------------
Extracts HuBERT hidden layer features from audio.
Supports optional TorchDR PCA dimensionality reduction.
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, HubertModel
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
# Arguments
# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="HuBERT feature extraction for Kaldi",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "input",
    type=str,
    help="Input: scp:path/to/wav.scp or ark:- for stdin"
)
parser.add_argument(
    "output",
    type=str,
    help="Output: ark:path/to/output.ark or ark:- for stdout"
)
parser.add_argument(
    "-l", "--layer",
    type=int,
    default=12,
    help="HuBERT encoder layer to extract (e.g., 12)",
)

parser.add_argument(
    "--write-utt2dur",
    "-wud",
    type=str,
    default=None,
    help="Optional utt2dur output: ark,t:path/to/utt2dur or ark:path/to/utt2dur",
)

args = parser.parse_args()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ------------------------------------------------------------------------- #
# Load HuBERT model
# ------------------------------------------------------------------------- #
logger.info(f"Loading HuBERT checkpoint")
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "facebook/hubert-base-ls960"
    )
    model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960"
    )
    model.to(device)
    model.eval()
   
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"HuBERT model loaded: {total_params:,} parameters")
except Exception as e:
    logger.error(f"Failed to load HuBERT model: {e}", exc_info=True)
    sys.exit(1)


def preprocess_waveform(waveform: np.ndarray) -> torch.Tensor:

    waveform = waveform.astype(np.float32)
    waveform /= np.iinfo(np.int16).max
    return torch.from_numpy(waveform).unsqueeze(0)

@torch.no_grad()
def compute_hubert(waveform: torch.Tensor, sample_rate: int = 16000):
    inputs = feature_extractor(
        waveform.squeeze(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs, output_hidden_states=True)
    feats = outputs.hidden_states[args.layer].squeeze(0)  
    
    return feats.cpu().numpy()

def calculate_duration(num_samples: int, sample_rate: int) -> float:
    return float(num_samples) / float(sample_rate)

# ------------------------------------------------------------------------- #
# Main processing
# ------------------------------------------------------------------------- #
def process_features():
    logger.info("=" * 70)
    logger.info(f"Input:        {args.input}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Layer:        {args.layer}")
    logger.info("=" * 70)

    utt2dur_data = {}
    processed = 0
    failed = 0

    if args.input == "ark:-":
        logger.info("Reading input from ark:- (stdin)")
        with ReadHelper("ark:-") as reader, WriteHelper(args.output) as writer:
            for utt_id, waveform in reader:
                try:
                    # waveform is assumed 1-D np.ndarray or similar
                    if isinstance(waveform, np.ndarray) and waveform.ndim == 1:
                        wf_t = torch.from_numpy(
                            waveform.astype(np.float32)
                        ).unsqueeze(0)
                    else:
                        wf_t = torch.tensor(
                            waveform, dtype=torch.float32
                        ).unsqueeze(0)

                    print(wf_t)
                    sys.exit(0)
                    feats = compute_hubert(wf_t, 16000)
                        

                    writer(utt_id, feats)
                    processed += 1
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue
    else:
        with ReadHelper(args.input) as reader, WriteHelper(args.output) as writer:
            for utt_id, (sample_rate, waveform) in reader:
                try:
                    wf_t = preprocess_waveform(waveform)
                    duration = calculate_duration(waveform.shape[0], sample_rate)
                    utt2dur_data[utt_id] = duration
                    
                    feats = compute_hubert(wf_t, 16000)

                    writer(utt_id, feats)
                    processed += 1

                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} utterances...")
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue

    logger.info("=" * 70)
    logger.info(f"Successfully processed: {processed} utterances")
    if failed > 0:
        logger.warning(f"Failed: {failed} utterances")
    logger.info("=" * 70)

    return utt2dur_data


def write_utt2dur(utt2dur_data: dict):
    if not args.write_utt2dur:
        return
    out_spec = args.write_utt2dur
    if out_spec.startswith("ark,t:"):
        path = out_spec[6:]
    elif out_spec.startswith("ark:"):
        path = out_spec[4:]
    else:
        path = out_spec

    logger.info(f"Writing utt2dur to: {path}")
    try:
        with open(path, "w") as f:
            for utt_id, dur in sorted(utt2dur_data.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur_data)} durations")
    except Exception as e:
        logger.error(f"Failed to write utt2dur: {e}")


# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        utt2dur = process_features()
        write_utt2dur(utt2dur)
        logger.info("HuBERT feature extraction completed successfully!")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
