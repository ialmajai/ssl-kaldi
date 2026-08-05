"""Command line entry point."""
import argparse
import logging
import os
import sys

from kaldiio import WriteHelper

from .extractor import SSLExtractor
from .io import read_file_list, read_wav_scp, write_side_files

logger = logging.getLogger("ssl_kaldi_feats")
logger.setLevel(logging.INFO)


def build_parser():
    p = argparse.ArgumentParser(
        prog="ssl-kaldi-feats",
        description="Kaldi ark/scp features from any HuggingFace speech encoder. "
                    "No Kaldi installation required.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model id, e.g. facebook/hubert-base-ls960")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--layer", type=int, help="single layer to extract, 1-indexed")
    g.add_argument("--layers",
                   help="comma-separated layers from ONE forward pass, e.g. "
                        "'10,12,14'. Costs one pass to the deepest layer "
                        "instead of one pass each")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav-scp", help="Kaldi wav.scp, including the 'cmd |' form")
    src.add_argument("--file-list", help="text file of audio paths, one per line")
    p.add_argument("--output-dir", required=True,
                   help="written as feats.scp/feats.ark plus utt2num_frames, "
                        "utt2dur and frame_shift. With --layers, one "
                        "subdirectory per layer")
    p.add_argument("--final-layer-norm", choices=["keep", "strip"], default="keep",
                   help="'keep' returns layer_norm(layer_k); 'strip' returns "
                        "hidden_states[k], which is what published layer studies "
                        "mean. Only affects models that norm after the layer "
                        "loop (the large ones, and Whisper)")
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    # argparse.BooleanOptionalAction would be tidier but is 3.9+, and this
    # package supports 3.8. Both halves carry help text so neither reads as an
    # accident.
    p.add_argument("--compress", dest="compress", action="store_true",
                   default=True,
                   help="write compressed archives, Kaldi kSpeechFeature, the "
                        "same codec as copy-feats --compress=true (the default)")
    p.add_argument("--no-compress", dest="compress", action="store_false",
                   help="write uncompressed archives. Compression is lossy, so "
                        "use this when comparing archives bit-for-bit")
    p.add_argument("--max-failures", type=int, default=0,
                   help="how many utterances may fail before aborting. A "
                        "tolerated failure silently shrinks the output")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else args.layer)
    extractor = SSLExtractor(args.model, layers, args.final_layer_norm, args.device)
    multi = len(extractor.layers) > 1

    dirs = {}
    for layer in extractor.layers:
        d = os.path.join(args.output_dir, f"layer{layer}") if multi else args.output_dir
        os.makedirs(d, exist_ok=True)
        dirs[layer] = d

    reader = (read_wav_scp(args.wav_scp) if args.wav_scp
              else read_file_list(args.file_list))

    comp = 2 if args.compress else None
    writers = {}
    for layer, d in dirs.items():
        spec = f"ark,scp:{os.path.abspath(d)}/feats.ark,{os.path.abspath(d)}/feats.scp"
        writers[layer] = WriteHelper(spec, compression_method=comp)

    utt2num_frames, utt2dur, failed = {}, {}, []
    processed = 0
    try:
        for utt_id, wav, sr in reader:
            try:
                feats = extractor.extract(wav, sr)
                for layer, w in writers.items():
                    w(utt_id, feats[layer])
                first = feats[extractor.layers[0]]
                utt2num_frames[utt_id] = len(first)
                utt2dur[utt_id] = len(wav) / float(sr)
                processed += 1
                if processed % 100 == 0:
                    logger.info("processed %d utterances", processed)
            except Exception as e:  # noqa: BLE001
                logger.error("failed on %s: %s", utt_id, e)
                failed.append(utt_id)
                if len(failed) > args.max_failures:
                    raise SystemExit(
                        f"aborting after {len(failed)} failure(s), "
                        f"--max-failures is {args.max_failures}. "
                        f"Failed: {' '.join(failed)}"
                    )
    finally:
        for w in writers.values():
            w.close()

    if processed == 0:
        raise SystemExit("no utterances were processed")

    for layer, d in dirs.items():
        write_side_files(d, utt2num_frames, utt2dur, extractor.frame_shift)

    logger.info("wrote %d utterances to %s", processed, args.output_dir)
    if failed:
        logger.warning("tolerated %d failure(s): %s", len(failed), " ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
