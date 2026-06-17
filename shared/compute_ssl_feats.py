#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import sys
import logging
from kaldiio import ReadHelper, WriteHelper
import argparse
from transformers import AutoFeatureExtractor, AutoModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SSLExtractor:
    def __init__(self, model_id, layer):
        """Loads the model and feature extractor exactly once into memory."""
        logger.info(f"Initializing model: {model_id}...")
        self.model_id = model_id
        self.layer = layer
        self.extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.encoder.layers = self.model.encoder.layers[:layer]  # fix 5: use param not global
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def extract(self, waveform, utt_id, target_layer=-1):
        try:
            inputs = self.extractor(waveform, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            all_layers = outputs.hidden_states
            total_layers = len(all_layers)

            # fix 2: raise so return stays inside try and caller gets a real error
            if target_layer >= total_layers or target_layer < -total_layers:
                raise ValueError(
                    f"Target layer {target_layer} out of bounds "
                    f"(model has {total_layers} layers after truncation)"
                )

            return all_layers[target_layer].cpu().squeeze(0).numpy()  # fix 2: return inside try

        except Exception as e:
            logger.error(f"Failed to process {utt_id}: {str(e)}")
            raise  # fix 2: re-raise so caller can count/skip


def preprocess_waveform(waveform: np.ndarray) -> np.ndarray:
    # fix 1: returns normalised float32 numpy array usable by both branches
    return waveform.astype(np.float32) / np.iinfo(np.int16).max


def calculate_duration(num_samples: int, sample_rate: int) -> float:
    return float(num_samples) / float(sample_rate)


def process_features():
    logger.info("=" * 70)
    logger.info(f"Input:        {args.input}")
    logger.info(f"Output:       {args.output}")
    logger.info(f"Layer:        {args.layer}")
    logger.info("=" * 70)

    ssl_extractor = SSLExtractor(args.ssl_model, args.layer)  # fix 5: pass layer as param

    utt2dur_data = {}
    processed = 0
    failed = 0

    if args.input == "ark:-":
        logger.info("Reading input from ark:- (stdin)")
        with ReadHelper("ark:-") as reader, WriteHelper(args.output) as writer:
            for utt_id, waveform in reader:
                try:
                    waveform = preprocess_waveform(waveform)  # fix 1: normalise ark:- path
                    utt2dur_data[utt_id] = waveform.shape[0] / 16000.0  # fix 3: populate utt2dur
                    feats = ssl_extractor.extract(waveform.squeeze(), utt_id, target_layer=args.layer)
                    writer(utt_id, feats)
                    processed += 1
                    if processed % 100 == 0:  # fix 4: progress logging
                        logger.info(f"Processed {processed} utterances...")
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue
    else:
        with ReadHelper(args.input) as reader, WriteHelper(args.output) as writer:
            for utt_id, (sample_rate, waveform) in reader:
                try:
                    utt2dur_data[utt_id] = calculate_duration(waveform.shape[0], sample_rate)
                    waveform = preprocess_waveform(waveform)
                    feats = ssl_extractor.extract(waveform.squeeze(), utt_id, target_layer=args.layer)
                    writer(utt_id, feats)
                    processed += 1
                    if processed % 100 == 0:
                        logger.info(f"Processed {processed} utterances...")
                except Exception as e:
                    logger.warning(f"Failed to process {utt_id}: {e}")
                    failed += 1
                    continue

    logger.info("=" * 70)
    logger.info(f"Successfully processed: {processed} utterances")
    if failed > 0:
        logger.warning(f"Failed: {failed} utterances")
    logger.info("=" * 70)

    return utt2dur_data


def write_utt2dur(utt2dur_data: dict):
    if not args.write_utt2dur:
        return
    out_spec = args.write_utt2dur
    if out_spec.startswith("ark,t:"):
        path = out_spec[6:]
    elif out_spec.startswith("ark:"):
        path = out_spec[4:]
    else:
        path = out_spec

    logger.info(f"Writing utt2dur to: {path}")
    try:
        with open(path, "w") as f:
            for utt_id, dur in sorted(utt2dur_data.items()):
                f.write(f"{utt_id} {dur:.3f}\n")
        logger.info(f"Wrote {len(utt2dur_data)} durations")
    except Exception as e:
        logger.error(f"Failed to write utt2dur: {e}")


if __name__ == "__main__":
    try:
        utt2dur = process_features()
        write_utt2dur(utt2dur)
        logger.info("SSL feature extraction completed successfully!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
