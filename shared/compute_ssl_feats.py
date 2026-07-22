#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys
import logging
from kaldiio import ReadHelper, WriteHelper
import argparse
from transformers import AutoFeatureExtractor, AutoModel

from kaldi_io_utils import write_utt2dur


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_SAMPLE_RATE = 16000


def parse_args():
    parser = argparse.ArgumentParser(
        description="SSL feature extraction for Kaldi",
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
        help="SSL embedding layer to extract (e.g., 12)",
    )
    parser.add_argument(
        "--write-utt2dur",
        "-wud",
        type=str,
        default=None,
        help="Optional utt2dur output: ark,t:path/to/utt2dur or ark:path/to/utt2dur",
    )
    parser.add_argument(
        "--ssl-model",
        "-model",
        type=str,
        default="facebook/hubert-base-ls960",
        help="Pretrained SSL model type from HuggingFace",
    )
    return parser.parse_args()


class SSLExtractor:
    def __init__(self, model_id, layer):
        """Loads the model and feature extractor exactly once into memory."""
        logger.info(f"Initializing model: {model_id}...")
        self.model_id = model_id
        self.layer = layer
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        total_layers = len(self.model.encoder.layers)
        if layer < 1 or layer > total_layers:
            raise ValueError(
                f"Layer {layer} out of bounds (model has {total_layers} layers)"
            )
        self.model.encoder.layers = self.model.encoder.layers[:layer]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def extract(self, waveform):
        inputs = self.extractor(
            waveform, sampling_rate=EXPECTED_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states = embeddings + one entry per remaining encoder layer,
        # so the last entry is the requested layer after truncation.
        return outputs.hidden_states[-1].cpu().squeeze(0).numpy()


def preprocess_waveform(waveform: np.ndarray) -> np.ndarray:
    if np.issubdtype(waveform.dtype, np.integer):
        return waveform.astype(np.float32) / np.iinfo(waveform.dtype).max
    return waveform.astype(np.float32)


def process_features(args):
    logger.info("=" * 70)
    logger.info(f"Input:        {args.input}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Layer:        {args.layer}")
    logger.info("=" * 70)

    ssl_extractor = SSLExtractor(args.ssl_model, args.layer)

    utt2dur_data = {}
    processed = 0
    failed = 0

    # kaldiio yields (utt_id, (sample_rate, waveform)) for wav input,
    # whether read from an scp or from an ark on stdin.
    with ReadHelper(args.input) as reader, WriteHelper(args.output) as writer:
        for utt_id, (sample_rate, waveform) in reader:
            try:
                if sample_rate != EXPECTED_SAMPLE_RATE:
                    raise ValueError(
                        f"sample rate {sample_rate} != {EXPECTED_SAMPLE_RATE}; "
                        "resample the audio first"
                    )
                dur = waveform.shape[0] / float(sample_rate)
                waveform = preprocess_waveform(waveform)
                feats = ssl_extractor.extract(waveform.squeeze())
                writer(utt_id, feats)
                utt2dur_data[utt_id] = dur
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Processed {processed} utterances...")
            except Exception as e:
                logger.warning(f"Failed to process {utt_id}: {e}")
                failed += 1

    logger.info("=" * 70)
    logger.info(f"Successfully processed: {processed} utterances")
    if failed > 0:
        logger.warning(f"Failed: {failed} utterances")
    logger.info("=" * 70)

    if processed == 0:
        logger.error("No utterances were successfully processed")
        sys.exit(1)

    return utt2dur_data


if __name__ == "__main__":
    args = parse_args()
    try:
        utt2dur = process_features(args)
        write_utt2dur(utt2dur, args.write_utt2dur)
        logger.info("SSL feature extraction completed successfully!")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
