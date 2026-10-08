#!/usr/bin/env bash
# Refine one utterance's word confidences from raw MBR posteriors (what
# kaldi_decode_e2e.sh prints) to calibrated probabilities, via
# steps/conf/apply_calibration.sh (logistic regression over
# [logit(raw posterior), log(word length in phones), log(unigram prob),
# word-identity], trained on real correct/incorrect labels -- see
# NOTES.md's confidence-calibration entry). Reuses the lattice
# kaldi_decode_e2e.sh already produced (exp/chain/.../decode_demo/lat.1.gz)
# rather than redecoding -- this script is *only* the confidence
# refinement, ~1-2s of extra lattice-rescale/depth/logistic-regression
# work, meant to be called after the transcript from kaldi_decode_e2e.sh
# is already on screen, not before.
#
# Usage: local/demo/kaldi_calibrate_conf.sh <utt-id>
#   Must be called after kaldi_decode_e2e.sh for the same utt-id, in the
#   same demo process -- relies on that decode's lattice still being at
#   $decode_dir (safe under this demo's single-request-at-a-time Gradio
#   queue; see demo.queue(default_concurrency_limit=1) in demo.py).
set -euo pipefail
cd "$(dirname "$0")/../.."  # egs/grid/s5
. ./cmd.sh
. ./path.sh

utt_id=$1

chain=exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x
# The trained calibration model is read-only repo content, always under
# $chain regardless of scratch location. Only the per-request decode
# output (produced by kaldi_decode_e2e.sh, same env var there) moves.
scratch=${KALDI_DEMO_SCRATCH:-$chain}
decode_dir=$scratch/decode_demo
# feats_dir is an unused positional arg in apply_calibration.sh (checked
# its source -- $data is never referenced past the argument list), so any
# existing directory satisfies it without a real dependency.
feats_dir=$scratch

rm -rf "$decode_dir/calib"
steps/conf/apply_calibration.sh --cmd "$decode_cmd" \
  "$feats_dir" data/lang "$decode_dir" "$chain/calibration" \
  "$decode_dir/calib" >&2

awk -v utt="$utt_id" '$1==utt {print $5, $6}' "$decode_dir/calib/ctm_calibrated"
