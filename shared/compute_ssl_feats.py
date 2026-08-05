#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import contextlib
import logging
from kaldiio import ReadHelper, WriteHelper
import argparse

# The extractor lives in packages/ssl-kaldi-feats so that it is usable without
# Kaldi and without this repo. It was duplicated here until 2026-08-05; the two
# copies were bit-exact at the point of merging. Install with:
#   pip install -e packages/ssl-kaldi-feats
from ssl_kaldi_feats import (  # noqa: F401
    EXPECTED_SAMPLE_RATE,
    WHISPER_SAMPLES_PER_FRAME,
    WHISPER_WINDOW_SECONDS,
    SSLExtractor,
    preprocess_waveform,
)

from kaldi_io_utils import write_utt2dur


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Set the level on our own logger, not just via basicConfig. Loading a model
# with AutoModel.from_pretrained() resets the ROOT logger to WARNING, which
# silently swallows every INFO we emit afterwards: progress counts, the
# layer_norm mode, and the final "processed N utterances". Only ERROR and
# WARNING survived, which is why extraction logs looked truncated at
# "Initializing model". A logger with its own level is immune.
logger.setLevel(logging.INFO)

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
        "--layers",
        type=str,
        default=None,
        help="comma-separated layers to extract in ONE forward pass, e.g. "
             "'12,16,18'. A sweep done this way costs one pass to the deepest "
             "layer instead of one pass per layer (18 layer-evaluations "
             "instead of 46 for that example). Requires --out-template and "
             "ignores the positional output argument",
    )
    parser.add_argument(
        "--out-template",
        type=str,
        default=None,
        help="wspecifier containing {layer}, used with --layers, e.g. "
             "'ark,scp:d/L{layer}.ark,d/L{layer}.scp'",
    )
    parser.add_argument(
        "--write-utt2num-frames",
        type=str,
        default=None,
        help="path for utt2num_frames, multi-layer mode only (the frame count "
             "is the same for every layer)",
    )
    parser.add_argument(
        "--write-utt2dur",
        "-wud",
        type=str,
        default=None,
        help="Optional utt2dur output: ark,t:path/to/utt2dur or ark:path/to/utt2dur",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="how many utterances may fail before the job aborts. A tolerated "
             "failure silently shrinks the data directory, which later looks "
             "like a smaller test set rather than an error",
    )
    parser.add_argument(
        "--ssl-model",
        "-model",
        type=str,
        default="facebook/hubert-base-ls960",
        help="Pretrained SSL model type from HuggingFace",
    )
    parser.add_argument(
        "--final-layer-norm",
        choices=["keep", "strip"],
        default="keep",
        help="what to do with the encoder-final layer_norm on models that apply "
             "it after the layer loop (do_stable_layer_norm=True, and Whisper). "
             "Truncating the stack moves that norm onto the requested layer. "
             "'keep' returns layer_norm(layer_k), so the features do NOT equal "
             "hidden_states[k] on the full model and the layer index is not "
             "comparable with the literature; 'strip' bypasses it, making "
             "truncation exactly equivalent to indexing. keep is the default "
             "because it measures better: on TIMIT mms-300m L14 it wins at "
             "every GMM stage by 0.67 to 2.21 PER (p<=0.15%%), the layer_norm "
             "acting as a free per-frame normalisation ahead of PCA. Use strip "
             "when you need features comparable with published layer studies. "
             "No effect on models that norm before the loop (hubert-base, "
             "wavlm-base, mHuBERT-147)",
    )
    return parser.parse_args()


def process_features(args):
    multi = args.layers is not None
    if multi and not args.out_template:
        raise SystemExit("--layers requires --out-template")
    layers = ([int(x) for x in args.layers.split(",")] if multi else args.layer)

    logger.info("=" * 70)
    logger.info(f"Input:        {args.input}")
    logger.info(f"Output:       {args.out_template if multi else args.output}")
    logger.info(f"Layer(s):     {layers}")
    logger.info("=" * 70)

    ssl_extractor = SSLExtractor(args.ssl_model, layers, args.final_layer_norm)

    utt2dur_data = {}
    utt2num_frames = {}
    processed = 0
    failed_utts = []

    # Single layer streams to one wspecifier, as before. Several layers cannot
    # share one stdout, so each gets its own compressed archive.
    with contextlib.ExitStack() as stack:
        reader = stack.enter_context(ReadHelper(args.input))
        if multi:
            writers = {
                L: stack.enter_context(
                    WriteHelper(args.out_template.format(layer=L),
                                compression_method=2)
                )
                for L in ssl_extractor.layers
            }
        else:
            writers = {ssl_extractor.layers[0]:
                       stack.enter_context(WriteHelper(args.output))}

        # kaldiio yields (utt_id, (sample_rate, waveform)) for wav input,
        # whether read from an scp or from an ark on stdin.
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
                for L, w in writers.items():
                    w(utt_id, feats[L])
                # Every layer has the same frame count, so one entry serves all.
                utt2num_frames[utt_id] = len(feats[ssl_extractor.layers[0]])
                utt2dur_data[utt_id] = dur
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Processed {processed} utterances...")
            except Exception as e:
                logger.error(f"Failed to process {utt_id}: {e}")
                failed_utts.append(utt_id)
                if len(failed_utts) > args.max_failures:
                    raise SystemExit(
                        f"aborting after {len(failed_utts)} failure(s), "
                        f"--max-failures is {args.max_failures}. "
                        f"Failed: {' '.join(failed_utts)}"
                    )

    logger.info("=" * 70)
    logger.info(f"Successfully processed: {processed} utterances")
    if failed_utts:
        logger.warning(
            f"Failed: {len(failed_utts)} utterances, tolerated because "
            f"--max-failures is {args.max_failures}: {' '.join(failed_utts)}"
        )
    logger.info("=" * 70)

    if processed == 0:
        logger.error("No utterances were successfully processed")
        sys.exit(1)

    if args.write_utt2num_frames:
        path = args.write_utt2num_frames
        for prefix in ("ark,t:", "ark:"):
            if path.startswith(prefix):
                path = path[len(prefix):]
                break
        with open(path, "w") as f:
            for utt_id, n in sorted(utt2num_frames.items()):
                f.write(f"{utt_id} {n}\n")
        logger.info(f"Wrote {len(utt2num_frames)} frame counts to {path}")

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
