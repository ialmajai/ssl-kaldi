#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)
# Apache 2.0

import argparse
import logging
import torch
import numpy as np
from kaldiio import ReadHelper, WriteHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("IPCA")


def _flush_batch(batch, ipca):
    batch_tensor = torch.from_numpy(np.vstack(batch)).float().to(ipca.device)
    ipca.partial_fit(batch_tensor)


def train_pca(feats_scp, ipca, max_utts):
    processed = 0
    batch = []
    with ReadHelper(f"scp:{feats_scp}") as reader:
        for utt, feats in reader:
            batch.append(feats)
            processed += 1
            if max_utts > 0 and processed >= max_utts:
                _flush_batch(batch, ipca)
                batch = []
                break
            if len(batch) >= 100:
                _flush_batch(batch, ipca)
                batch = []
                logger.info(f"[train] processed {processed}")

    if batch:
        _flush_batch(batch, ipca)

    logger.info(f"[train] done: processed={processed}")


def save_pca(ipca, path):
    # Save only what transform() needs, as CPU tensors, so the model can be
    # loaded on any host and any torchdr version.
    torch.save(
        {
            "components": ipca.components_.detach().float().cpu(),
            "mean": ipca.mean_.detach().float().cpu(),
        },
        path,
    )


def load_pca(path, device):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Legacy format: a pickled torchdr IncrementalPCA object.
        logger.info("Falling back to legacy pickled PCA model format")
        ipca = torch.load(path, map_location="cpu", weights_only=False)
        state = {
            "components": ipca.components_.detach().float().cpu(),
            "mean": ipca.mean_.detach().float().cpu(),
        }
    return state["components"].to(device), state["mean"].to(device)


def apply_pca(feats_scp, writer_spec, components, mean):
    processed = 0

    with ReadHelper(f"scp:{feats_scp}") as reader, \
         WriteHelper(writer_spec) as writer:
        for utt, feats in reader:
            feats_t = torch.from_numpy(feats).float().to(components.device)

            feats_pca = (feats_t - mean) @ components.T

            writer(utt, feats_pca.cpu().numpy())
            processed += 1

            if processed % 100 == 0:
                logger.info(f"[apply] processed {processed}")

    logger.info(f"[apply] done: processed={processed}")


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
    parser.add_argument("--device", default=None,
                        help="torch device; defaults to cuda-if-available for "
                             "train and cpu for apply")
    parser.add_argument("output", type=str, nargs="?", default="ark:-")
    args = parser.parse_args()

    if args.mode == "train":
        from torchdr import IncrementalPCA

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        ipca = IncrementalPCA(
            n_components=args.pca_dim,
            device=device
        )
        logger.info("Training PCA…")
        train_pca(args.feats_scp, ipca, args.max_utts)
        save_pca(ipca, args.pca_model)
        logger.info(f"PCA model saved to {args.pca_model}")

    elif args.mode == "apply":
        device = args.device or "cpu"
        logger.info(f"Using device: {device}")
        logger.info("Applying PCA…")
        components, mean = load_pca(args.pca_model, device)
        if components.shape[0] != args.pca_dim:
            logger.warning(
                f"--pca_dim={args.pca_dim} ignored: model has "
                f"{components.shape[0]} components"
            )
        apply_pca(args.feats_scp, args.output, components, mean)


if __name__ == "__main__":
    main()
