#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

import argparse
import logging
import torch
from kaldiio import ReadHelper, WriteHelper
import torch.nn.functional as F

# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Upsampling")


def interpolate(feats_scp, writer_spec, interp_mode, upsample_factor, device):    

    processed = 0

    with ReadHelper(f"scp:{feats_scp}") as reader, \
         WriteHelper(writer_spec) as writer:
        for utt, feats in reader:
            feats = torch.from_numpy(feats).to(device)            
            
            x = feats.transpose(0, 1).unsqueeze(0)

            x_up = F.interpolate(
                x,
                scale_factor=upsample_factor,
                mode=interp_mode,
                align_corners=False if interp_mode == "linear" else None
            )

            writer(utt, x_up.squeeze(0).transpose(0, 1).cpu().numpy())
            processed += 1

            if processed % 100 == 0:
                logger.info(f"[apply] processed {processed}")

    logger.info(f"[apply] done: processed={processed}")
    
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Interpolate Kaldi features"
    )
    parser.add_argument("--feats_scp", required=True)
    parser.add_argument("--interp_mode", default="linear",
                        choices=["linear", "nearest"],
                        help="Interpolation mode for upsampling")
    parser.add_argument("--upsample_factor",    type=int,    default=2, 
                        help="Temporal upsampling factor (e.g. 2 = double frame_rate)")
    parser.add_argument("output", type=str, default="ark:-")
    args = parser.parse_args()

    device = "cpu"
    logger.info(f"Using device: {device}")
     
    interpolate(args.feats_scp, args.output, args.interp_mode,
                  args.upsample_factor, device)

    logger.info("All done.")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()

