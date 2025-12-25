#!/usr/bin/env python3
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

import argparse
import subprocess
import numpy as np
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from transformers import HubertModel, Wav2Vec2FeatureExtractor, HubertConfig
from torchdr import IncrementalPCA
import io
import torchaudio
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("IPCA")


def load_audio(entry: str): 

    entry = entry.strip()
 
    if entry.endswith("|"):
        cmd = entry[:-1].strip()

        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        wav_bytes = proc.stdout.read()
        proc.wait()

        if len(wav_bytes) < 44:
            raise ValueError("Invalid audio from pipe, too short")

        bio = io.BytesIO(wav_bytes)
        audio, sr = torchaudio.load(bio)
        return audio, sr

    audio, sr = torchaudio.load(entry)
    return audio, sr

# =====================================================================
#   HuBERT feature extraction
# =====================================================================
def extract_hubert(waveform: torch.Tensor, model, extractor, layer: int):
    with torch.no_grad():   
        inputs = extractor(
            waveform.squeeze(),
            sampling_rate=16000,
            return_tensors="pt"
        ).to(waveform.device)

        output = model(**inputs, output_hidden_states=True)

    return output.hidden_states[layer].squeeze(0)

class KaldiWavScpDataset(Dataset):
    def __init__(self, wav_scp: str, layer: int, max_files: int):
        self.layer = layer
        self.entries = []

        with open(wav_scp, "r") as f:
            for line in f:
                utt, cmd_or_path = line.strip().split(maxsplit=1)                
                self.entries.append((utt, cmd_or_path))

        if max_files > 0:
            self.entries = self.entries[:max_files]

        logger.info(f"Loaded {len(self.entries)} wav.scp entries")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        utt, entry = self.entries[idx]

        try:
            audio, sr = load_audio(entry)        
            audio = audio.unsqueeze(0) if audio.ndim == 1 else audio  # ensure [1, T]
            return {"waveform": audio, "utt": utt, "success": True}
        except Exception as e:
            logger.warning(f"[{utt}] failed: {e}")
            return {"waveform": None, "utt": utt, "success": False}


# =====================================================================
#   MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str,required=True)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--max_files", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="pca_hubert")
    parser.add_argument("--output_model", type=str, default="ipca_torchdr.pt")
    args = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load HuBERT 
    logger.info("Loading HuBERT model…")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

    config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
    config.output_hidden_states = True
    config.num_hidden_layers = args.layer  

    model = HubertModel(config).to(device).eval()

    # Dataset + DataLoader
    ds = KaldiWavScpDataset(args.wav_scp, args.layer, args.max_files)
    torch.manual_seed(2809)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.workers,
        shuffle=True,
        collate_fn=lambda x: x, 
        pin_memory=True
    )

    ipca = IncrementalPCA(
        n_components=args.pca_dim,
        device=device
    )

    logger.info(f"Starting Incremental PCA with {args.pca_dim} dims")

    # ---------------------------------------------------------------
    # Streaming PCA
    # ---------------------------------------------------------------
    for batch_list in loader: 
        batch_tensors = [] 
        for sample in batch_list:
            if not sample["success"]:
                continue
            waveform = sample["waveform"][0].to(device)
            feats = extract_hubert(waveform, model, extractor, args.layer)

            ipca.partial_fit(feats.to(device))
            
            batch_tensors.append(feats)
        ipca.partial_fit(torch.vstack(batch_tensors).to(device))
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # Save PCA model
    # ---------------------------------------------------------------
    logger.info(f"Saving IPCA to {args.output_dir}/{args.output_model} …")
    torch.save(ipca, args.output_dir + "/" + args.output_model)

    logger.info("Done.")


if __name__ == "__main__":
    main()

