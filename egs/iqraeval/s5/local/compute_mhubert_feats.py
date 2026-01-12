#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0
"""
mHuBERT Feature Extraction for Kaldi
"""

import sys
import argparse
import logging
import numpy as np
import torch
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
    help="Encoder layer to extract features"
)
parser.add_argument(
    "--write-utt2dur",
    type=str,
    default=None,
    help="Output duration file: ark,t:path/to/utt2dur"
)

args = parser.parse_args()

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
    model.encoder.layers = model.encoder.layers[:args.layer]
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)
    
# ------------------------------------------------------------------------- #
# Feature Extraction Functions
# ------------------------------------------------------------------------- #
@torch.no_grad()
def compute_mhubert(waveform: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract mHuBERT features from waveform.
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
   
    return features.cpu().numpy()

def calculate_duration(num_samples: int, sample_rate: int) -> float:
    """Calculate duration in seconds."""
    return float(num_samples) / sample_rate

# ------------------------------------------------------------------------- #
# Main Processing
# ------------------------------------------------------------------------- #
def main():
    """Main feature extraction pipeline."""
    
    logger.info("=" * 70)
    logger.info(f"Input:  {args.input}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Layer:  {args.layer}")
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
        utt2dur_data = main()        
        write_utt2dur(utt2dur_data)        
        logger.info("Feature extraction completed successfully!")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)       
