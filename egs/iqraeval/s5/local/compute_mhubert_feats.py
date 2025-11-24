#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0
"""
mHuBERT Feature Extraction for Kaldi
----------------------------------------
Extracts mHuBERT hidden layer features from audio using Kaldi I/O.
Supports optional PCA dimensionality reduction.

Usage:
    # Extract full 768-dim features (no PCA)
    python extract_mhubert.py scp:data/train/wav.scp ark:exp/features/train.ark
    
    # With PCA reduction to 30-dim
    python extract_mhubert.py scp:data/train/wav.scp ark:exp/features/train.ark \
        --apply-pca --dim 30
    
    # With duration output
    python extract_mhubert.py scp:data/train/wav.scp ark:exp/features/train.ark \
        --write-utt2dur ark,t:data/train/utt2dur
"""

import sys
import argparse
import logging
import subprocess
import io
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from kaldiio import ReadHelper, WriteHelper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- #
# Arguments
# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Extract mHuBERT features for Kaldi",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    'input', 
    type=str,
    help='Input: scp:path/to/wav.scp or ark:- for stdin'
)
parser.add_argument(
    'output', 
    type=str,
    help='Output: ark:path/to/output.ark or ark:- for stdout'
)
parser.add_argument(
    "-l", "--layer", 
    type=int, 
    default=9,
    help="Encoder layer to extract (0-12)"
)
parser.add_argument(
    "-d", "--dim", 
    type=int, 
    default=None,  
    help="PCA feature dimension"
)
parser.add_argument(
    "--apply-pca", 
    action='store_true',
    help="Apply PCA dimensionality reduction (requires --dim)"
)
parser.add_argument(
    "--pca-dir",
    type=str,
    default="pca_mhubert",
    help="Directory containing PCA model"
)
parser.add_argument(
    "--write-utt2dur",
    type=str,
    default=None,
    help="Output duration file: ark,t:path/to/utt2dur"
)

args = parser.parse_args()

# Validate PCA arguments
if args.apply_pca and args.dim is None:
    logger.error("--apply-pca requires --dim to be specified")
    sys.exit(1)

# ------------------------------------------------------------------------- #
# Device Setup
# ------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ------------------------------------------------------------------------- #
# Load mHuBERT Model
# ------------------------------------------------------------------------- #
logger.info("Loading mHuBERT-147 model...")
try:
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
    model = HubertModel.from_pretrained("utter-project/mHuBERT-147")
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)
    



# ------------------------------------------------------------------------- #
# Load PCA Model (if needed)
# ------------------------------------------------------------------------- #
ipca = None
output_dim = 768  # Default: full mHuBERT dimension

if args.apply_pca:
    pca_path = Path(args.pca_dir) / f"pca-mhubert-l{args.layer}-{args.dim}d.pt"
    
    if not pca_path.exists():
        logger.error(f"PCA model not found: {pca_path}")
        logger.error("Run PCA training first or disable --apply-pca")
        sys.exit(1)
    
    logger.info(f"Loading PCA model from {pca_path}")
    try:
        ipca = torch.load(pca_path, map_location=device)
        ipca.device = device
        output_dim = args.dim
        logger.info(f"PCA enabled: 768 → {args.dim} dimensions")
    except Exception as e:
        logger.error(f"Failed to load PCA: {e}")
        sys.exit(1)
else:
    logger.info(f"PCA disabled: extracting full 768-dimensional features from layer {args.layer}")

# ------------------------------------------------------------------------- #
# Feature Extraction Functions
# ------------------------------------------------------------------------- #

@torch.no_grad()
def compute_mhubert(waveform: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract mHuBERT features from waveform.
    
    Args:
        waveform: Preprocessed waveform tensor [1, T]
        sample_rate: Sample rate
    
    Returns:
        Feature array [T, D] where:
        - D = 768 if PCA is disabled
        - D = args.dim if PCA is enabled
    """
    # Feature extraction
    inputs = feature_extractor(
        waveform.squeeze(), 
        sampling_rate=sample_rate, 
        return_tensors="pt"
    ).to(device)
    
    # Forward pass
    output = model(**inputs, output_hidden_states=True)
    features = output.hidden_states[args.layer].squeeze()  # [T, 768]
    # Apply PCA only if requested
    if args.apply_pca and ipca is not None:
        features = ipca.transform(features)
        features = features[:, :args.dim]  # Truncate to requested dimension
    # Otherwise return full 768-dim features
    return features.cpu().numpy()


def calculate_duration(num_samples: int, sample_rate: int) -> float:
    """Calculate duration in seconds."""
    return float(num_samples) / sample_rate

# ------------------------------------------------------------------------- #
# Main Processing
# ------------------------------------------------------------------------- #
def process_features():
    """Main feature extraction pipeline."""
    
    logger.info("=" * 70)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Layer:  {args.layer}")
    logger.info(f"Output dimension: {output_dim}")
    logger.info("=" * 70)
    
    # Storage for durations
    utt2dur_data = {}
    processed_count = 0
    failed_count = 0
    
    try:
        with ReadHelper(args.input) as reader, WriteHelper('ark:-') as writer:
            for utt_id, (sample_rate, waveform) in reader:
                try:                             
                    # Calculate duration
                    duration = calculate_duration(waveform.shape[0], sample_rate)
                    utt2dur_data[utt_id] = duration 
                    
                    # Extract features
                    features = compute_mhubert(torch.tensor(waveform), sample_rate)
                
                    # Write to ark
                    writer(utt_id, features)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} utterances...")
                
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


def write_utt2dur(utt2dur_data: dict):
    """Write utt2dur file if specified."""
    
    if not args.write_utt2dur:
        return
    
    # Parse output format
    output_path = args.write_utt2dur
    if output_path.startswith("ark,t:"):
        output_path = output_path[6:]
    elif output_path.startswith("ark:"):
        output_path = output_path[4:]
    
    logger.info(f"Writing duration file: {output_path}")
    
    try:
        with open(output_path, "w") as f:
            for utt_id, duration in sorted(utt2dur_data.items()):
                f.write(f"{utt_id} {duration:.3f}\n")
        logger.info(f"Wrote {len(utt2dur_data)} durations to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write utt2dur: {e}")

# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        # Extract features
        utt2dur_data = process_features()
        
        # Write durations if requested
        write_utt2dur(utt2dur_data)
        
        logger.info("✅ Feature extraction completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

       
