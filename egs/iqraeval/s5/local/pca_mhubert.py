#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0
"""
Incremental PCA on mHuBERT-147 Hidden States
--------------------------------------------
Extracts mHuBERT-147 hidden layer features from speech datasets and trains
an Incremental PCA model for dimensionality reduction.

Usage:
    python pca_mhubert.py --layer 12 --pca_dim 30
    python pca_mhubert.py --layer 9 --pca_dim 40 --num_files 5000
"""

import argparse
import random
import logging
from pathlib import Path
from glob import glob

import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from torchdr import IncrementalPCA

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------- #
# CLI Arguments
# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="Train Incremental PCA on mHuBERT-147 features for IqraEval"
)
parser.add_argument(
    "-l", "--layer", 
    type=int, 
    default=9, 
    help="mHuBERT encoder layer to extract"
)
parser.add_argument(
    "-d", "--pca_dim", 
    type=int, 
    default=30, 
    help="PCA output dimension"
)
parser.add_argument(
    "-n", "--num_files", 
    type=int, 
    default=3000, 
    help="Max wav files per dataset"
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=32, 
    help="DataLoader batch size"
)
parser.add_argument(
    "--workers", 
    type=int, 
    default=0, 
    help="DataLoader workers (default: 0 to avoid pickling issues)"
)
parser.add_argument(
    "--seed", 
    type=int, 
    default=42, 
    help="Random seed for reproducibility (default: 42)"
)
parser.add_argument(
    "--data_root", 
    type=str, 
    default="/data/git/interspeech_IqraEval/sws_data",
    help="Root directory containing datasets"
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    default="pca_mhubert",
    help="Output directory for PCA model (default: pca_mhubert)"
)

args = parser.parse_args()

# Setup
mp.set_start_method("spawn", force=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(args.seed)
torch.manual_seed(args.seed)

logger.info(f"Device: {DEVICE}")
logger.info(f"Settings: layer={args.layer}, pca_dim={args.pca_dim}, batch_size={args.batch_size}")

# ------------------------------------------------------------------------- #
# Load mHuBERT-147 Model
# ------------------------------------------------------------------------- #
logger.info("Loading mHuBERT-147 model from Hugging Face...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("utter-project/mHuBERT-147")
hubert_model = HubertModel.from_pretrained("utter-project/mHuBERT-147").to(DEVICE)
hubert_model.eval()

# Count parameters
total_params = sum(p.numel() for p in hubert_model.parameters())
logger.info(f"Model loaded: {total_params:,} parameters")

# ------------------------------------------------------------------------- #
# Feature Extraction Function
# ------------------------------------------------------------------------- #
@torch.no_grad()
def extract_features(audio_path: str, layer_idx: int) -> torch.Tensor:
    """
    Extract features from specified mHuBERT layer.
    
    Args:
        audio_path: Path to audio file
        layer_idx: Layer index to extract (0-12)
    
    Returns:
        Feature tensor of shape [time_steps, hidden_dim]
    """
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Extract features
    inputs = feature_extractor(
        waveform.squeeze(), 
        sampling_rate=16000, 
        return_tensors="pt"
    ).to(DEVICE)
    
    output = hubert_model(**inputs, output_hidden_states=True)
    features = output.hidden_states[layer_idx].squeeze(0).cpu()
    
    return features

# ------------------------------------------------------------------------- #
# Dataset
# ------------------------------------------------------------------------- #
class AudioDataset(Dataset):
    """Dataset for loading audio files and extracting mHuBERT features."""
    
    def __init__(self, root_dir: str, layer_idx: int, limit: int = -1, seed: int = 42):
        """
        Args:
            root_dir: Directory containing wav files
            layer_idx: mHuBERT layer to extract
            limit: Maximum number of files (-1 for all)
            seed: Random seed for shuffling
        """
        self.layer_idx = layer_idx
        
        # Collect all wav files
        self.filepaths = sorted(glob(f"{root_dir}/**/*.wav", recursive=True))
        
        # Shuffle with seed
        random.seed(seed)
        random.shuffle(self.filepaths)
        
        # Limit if specified
        if limit > 0:
            self.filepaths = self.filepaths[:limit]
        
        logger.info(f"Dataset: {len(self.filepaths)} files from {root_dir}")
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        path = self.filepaths[idx]
        try:
            features = extract_features(path, self.layer_idx)
            return {"features": features, "success": True}
        except Exception as e:
            logger.warning(f"Failed to process {path}: {e}")
            return {"features": None, "success": False}

def collate_fn(batch):
    """Collate function that filters failed samples and concatenates features."""
    valid = [item for item in batch if item["success"]]
    
    if not valid:
        return {"features": torch.empty(0, 0)}
    
    features = [item["features"] for item in valid]
    return {"features": torch.cat(features, dim=0)}

# ------------------------------------------------------------------------- #
# PCA Training
# ------------------------------------------------------------------------- #
def train_pca():
    """Train Incremental PCA on mHuBERT features from multiple datasets."""
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup datasets
    data_root = Path(args.data_root)
    datasets = {
        "TTS": data_root / "TTS" / "train" / "wav",
        "CV-Ar": data_root / "CV-Ar" / "train" / "wav"
    }
    
    # Create data loaders
    loaders = []
    for name, path in datasets.items():
        if not path.exists():
            logger.warning(f"Path not found: {path}, skipping {name}")
            continue
        
        ds = AudioDataset(
            root_dir=str(path),
            layer_idx=args.layer,
            limit=args.num_files,
            seed=args.seed
        )
        
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=(DEVICE.type == 'cuda'),
            drop_last=False
        )
        loaders.append((name, dl))
    
    if not loaders:
        raise ValueError("No valid datasets found!")
    
    # Initialize Incremental PCA
    logger.info(f"Initializing IncrementalPCA: {args.pca_dim} components")
    ipca = IncrementalPCA(n_components=args.pca_dim, device=DEVICE)
    
    # Train PCA
    total_frames = 0
    total_batches = sum(len(dl) for _, dl in loaders)
    
    logger.info("Starting PCA training...")
    with tqdm(total=total_batches, desc="Training PCA") as pbar:
        # Process all loaders in parallel (zip)
        for batch_tuple in zip(*[dl for _, dl in loaders]):
            for batch in batch_tuple:
                if batch["features"].numel() > 0:
                    ipca.partial_fit(batch["features"])
                    total_frames += batch["features"].shape[0]
                pbar.update(1)
    
    logger.info(f"PCA training complete: {total_frames:,} feature frames processed")
    
    # Save model
    save_path = output_dir / f"pca-mhubert-l{args.layer}-{args.pca_dim}d.pt"
    torch.save(ipca, save_path)
    logger.info(f"✅ PCA model saved: {save_path}")
    
    # Log explained variance
    if hasattr(ipca, 'explained_variance_ratio_'):
        total_var = ipca.explained_variance_ratio_.sum().item()
        logger.info(f"Explained variance: {total_var:.4f} ({total_var*100:.2f}%)")
        
        # Show per-component variance
        top_5_var = ipca.explained_variance_ratio_[:5]
        logger.info(f"Top 5 components: {top_5_var.tolist()}")
    
    return save_path

# ------------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    try:
        logger.info("=" * 70)
        logger.info("mHuBERT-147 Incremental PCA Training")
        logger.info("=" * 70)
        
        pca_path = train_pca()
        
        logger.info("=" * 70)
        logger.info("Training completed successfully!")
        logger.info(f"PCA model: {pca_path}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


