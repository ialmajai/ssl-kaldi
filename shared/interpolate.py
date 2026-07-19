#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)
# Apache 2.0

import argparse
import logging
import torch
import torch.nn.functional as F
from kaldiio import ReadHelper, WriteHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Upsampling")


def interpolate(feats_scp: str, writer_spec: str, interp_mode: str, upsample_factor: int, device: str) -> None:
    processed = 0
    interp_kwargs: dict = {"scale_factor": upsample_factor, "mode": interp_mode}
    if interp_mode == "linear":
        interp_kwargs["align_corners"] = False

    with ReadHelper(f"scp:{feats_scp}") as reader, WriteHelper(writer_spec) as writer:
        for utt, feats in reader:
            x = torch.from_numpy(feats).to(device).T.unsqueeze(0)
            x_up = F.interpolate(x, **interp_kwargs)
            writer(utt, x_up.squeeze(0).T.cpu().numpy())
            processed += 1
            if processed % 100 == 0:
                logger.info(f"processed {processed}")

    logger.info(f"done: processed={processed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolate Kaldi features")
    parser.add_argument("--feats_scp", required=True)
    parser.add_argument("--interp_mode", default="linear", choices=["linear", "nearest"])
    parser.add_argument("--upsample_factor", type=int, default=2)
    parser.add_argument("--device", default="cpu", help="torch device (cpu or cuda)")
    parser.add_argument("output", type=str, nargs="?", default="ark:-")
    args = parser.parse_args()

    logger.info(f"device={args.device} mode={args.interp_mode} factor={args.upsample_factor}")
    interpolate(args.feats_scp, args.output, args.interp_mode, args.upsample_factor, args.device)
    logger.info("All done.")


if __name__ == "__main__":
    main()
