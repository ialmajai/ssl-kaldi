#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

import argparse
import logging
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
def train_pca(feats_scp, ipca, max_utts):
    processed, skipped = 0, 0
    with ReadHelper(f"scp:{feats_scp}") as reader:
        batch = []
        count = 0
        for utt, feats in reader:
            count += 1
            batch.append(feats)            
            if count >= 500: 
                batch_tensor = torch.from_numpy(np.vstack(batch)).float().to(ipca.device)              
                ipca.partial_fit(batch_tensor)                       
                processed += 500
                batch = []
                count = 0
                logger.info(f"[train] processed {processed}")
                
            if max_utts > 0 and processed >= max_utts:
                if batch:
                    batch_tensor = torch.from_numpy(np.vstack(batch)).float().to(ipca.device)              
                    ipca.partial_fit(batch_tensor)
                break

    logger.info(f"[train] done: processed={processed}, skipped={skipped}")


def apply_pca(feats_scp, writer_spec, ipca):    

    processed = 0

    with ReadHelper(f"scp:{feats_scp}") as reader, \
         WriteHelper(writer_spec) as writer:
        for utt, feats in reader:
            feats_t = torch.from_numpy(feats).to(ipca.device)
            
            feats_pca = ipca.transform(feats_t)

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
    parser.add_argument("--pca_dim", type=int, default=30)
    parser.add_argument("--max_utts", type=int, default=-1)
    parser.add_argument("--pca_model", default="ipca.pt")
    parser.add_argument("output", type=str, default="ark:-")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --------------------------------------------------------
    # Train PCA
    # --------------------------------------------------------
    if args.mode == "train":
        ipca = IncrementalPCA(
            n_components=args.pca_dim,
            device=device
        )
        logger.info("Training PCA…")
        train_pca(args.feats_scp, ipca, args.max_utts)
        torch.save(ipca, args.pca_model)
        logger.info(f"PCA model saved to {args.pca_model}")

    # --------------------------------------------------------
    # Apply PCA
    # --------------------------------------------------------
    if args.mode == "apply":
        logger.info("Applying PCA…")
        ipca = torch.load(args.pca_model)
        apply_pca(args.feats_scp, args.output, ipca)

    logger.info("All done.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()

