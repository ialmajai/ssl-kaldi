#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

import argparse
import logging
import os
import torch
import numpy as np
from torchdr import IncrementalPCA
from kaldiio import ReadHelper, WriteHelper

# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("IPCA")

# ------------------------------------------------------------
def save_ipca(ipca, path):
    payload = {
        "components": ipca.components_.detach().cpu(),
        "mean": ipca.mean_.detach().cpu(),
        "var": ipca.var_.detach().cpu(),
        "n_samples": ipca.n_samples_seen_,
    }
    torch.save(payload, path)

def load_ipca(path, device):
    state = torch.load(path, map_location=device)
    ipca = IncrementalPCA(
        n_components=state["components"].shape[0],
        device=device
    )
    ipca.components_ = state["components"].to(device)
    ipca.mean_ = state["mean"].to(device)
    ipca.var_ = state["var"].to(device)
    ipca.n_samples_seen_ = state["n_samples"]
    return ipca

# ------------------------------------------------------------
def train_pca(feats_scp, ipca, pca_dim, max_utts):
    processed, skipped = 0, 0

    with ReadHelper(f"scp:{feats_scp}") as reader:
        for utt, feats in reader:
            if feats is None or feats.shape[0] == 0:
                skipped += 1
                continue

            if feats.shape[0] < pca_dim:
                skipped += 1
                continue

            feats_t = torch.from_numpy(feats).float().to(ipca.device)
            ipca.partial_fit(feats_t)

            processed += 1
            if max_utts > 0 and processed >= max_utts:
                break

            if processed % 100 == 0:
                logger.info(f"[train] processed {processed}")

    logger.info(f"[train] done: processed={processed}, skipped={skipped}")


def apply_pca(feats_scp, writer_spec, ipca):    

    processed = 0

    with ReadHelper(f"scp:{feats_scp}") as reader, \
         WriteHelper(writer_spec) as writer:

        for utt, feats in reader:
            feats_t = torch.from_numpy(feats).float().to(ipca.device)

            # PCA transform: (X - mean) @ components^T
            feats_pca = (feats_t - ipca.mean_) @ ipca.components_.t()

            writer(utt, feats_pca.cpu().numpy())
            processed += 1

            if processed % 100 == 0:
                logger.info(f"[apply] processed {processed}")

    logger.info(f"[apply] done: processed={processed}")

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train and/or apply PCA on Kaldi features"
    )
    parser.add_argument("--mode", choices=["train", "apply"],
                        required=True)
    parser.add_argument("--feats_scp", required=True)
    parser.add_argument("output", type=str, help="Output: ark:path/to/output.ark or ark:- for stdout")
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--max_utts", type=int, default=-1)
    parser.add_argument("--pca_model", default="ipca.pt")
    parser.add_argument("--output_dir", default="pca")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    pca_path = os.path.join(args.output_dir, args.pca_model)

    # --------------------------------------------------------
    # Train PCA
    # --------------------------------------------------------
    if args.mode == "train":
        ipca = IncrementalPCA(
            n_components=args.pca_dim,
            device=device
        )
        logger.info("Training PCA…")
        train_pca(args.feats_scp, ipca, args.pca_dim, args.max_utts)
        save_ipca(ipca, args.output)
        logger.info(f"PCA model saved to {pca_path}")

    # --------------------------------------------------------
    # Apply PCA
    # --------------------------------------------------------
    if args.mode == "apply":
        logger.info("Applying PCA…")
        ipca = load_ipca(pca_path, device)
        apply_pca(args.feats_scp, args.output, ipca)

    logger.info("All done.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()

