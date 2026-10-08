#!/usr/bin/env bash
# Decode one already-extracted, already-PCA'd-and-upsampled AV-HuBERT
# utterance (base model, layer 4, meanface, 30-dim, 50fps -- PCA and
# upsampling now done in-process by kaldi_lipreader.py's _pca_and_upsample,
# not by this script: shared/make_pca_features.sh + shared/upsample_features.sh
# are batch scripts built for thousands of utterances, and for one ~75-frame
# clip their fixed process-spawn overhead was ~4.85s out of a ~9.9s total
# decode latency for math that takes microseconds) through the recipe's best
# base-l4 system: GMM-SAT (for a per-utterance fMLLR transform) -> fMLLR
# TDNN-F chain. Prints the 1-best transcript to stdout.
#
# Usage: local/demo/kaldi_decode.sh <utt-id> <feats-dir>
#   <feats-dir> must contain feats.scp/utt2spk/spk2utt for exactly one
#   utterance (30-dim, 50fps, meanface-cropped base-layer-4-then-PCA'd
#   features -- i.e. already in the same space data/sw_l4_*_up is in).
set -euo pipefail
cd "$(dirname "$0")/../.."  # egs/grid/s5
. ./cmd.sh
. ./path.sh

utt_id=$1
up_dir=$2

gmm=exp-sweep-mf50/l4/tri3b
chain=exp/chain/tdnntdnnf_l4

# Per-request scratch outside the repo tree by default unset (bare-metal
# behavior, unchanged) -- Docker deployment sets KALDI_DEMO_SCRATCH to a
# writable tmpfs so the repo itself can be mounted read-only, same as
# kaldi_decode_e2e.sh/kaldi_calibrate_conf.sh.
scratch=${KALDI_DEMO_SCRATCH:-}
gmm_decode_dir=${scratch:-$gmm}/decode_demo
fmllr_dir=${scratch:-data-fmllr-tri3b}/demo
chain_decode_dir=${scratch:-$chain}/decode_demo
mkdir -p "${scratch:-.}"

stage() { echo "[stage] $(date +%H:%M:%S.%N | cut -c1-12) $1" >&2; }

stage "cmvn stats start"
steps/compute_cmvn_stats.sh "$up_dir" >&2
utils/fix_data_dir.sh "$up_dir" >&2

stage "gmm decode_fmllr start"
mkdir -p "$gmm_decode_dir"; find "$gmm_decode_dir" -mindepth 1 -delete
steps/decode_fmllr.sh --nj 1 --cmd "$decode_cmd" --skip-scoring true \
  "$gmm/graph" "$up_dir" "$gmm_decode_dir" >&2

stage "fmllr feature build start"
mkdir -p "$fmllr_dir"; find "$fmllr_dir" -mindepth 1 -delete
steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
  --transform-dir "$gmm_decode_dir" \
  "$fmllr_dir" "$up_dir" "$gmm" "$fmllr_dir/log" "$fmllr_dir/data" >&2
steps/compute_cmvn_stats.sh "$fmllr_dir" >&2
utils/fix_data_dir.sh "$fmllr_dir" >&2

stage "chain decode start"
mkdir -p "$chain_decode_dir"; find "$chain_decode_dir" -mindepth 1 -delete
steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
  --nj 1 --cmd "$decode_cmd" --skip-scoring true \
  "$chain/graph" "$fmllr_dir" "$chain_decode_dir" >&2

stage "confidence scoring start"

# Per-word confidence from lattice posteriors (Minimum-Bayes-Risk decoded),
# not just a 1-best transcript -- lets the UI colour-code words the way
# icefall's k2 arc-posterior confidences did. One "word conf" pair per
# stdout line, in sentence order; the caller reconstructs the transcript by
# joining the words.
lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=true \
  "ark:gunzip -c $chain_decode_dir/lat.1.gz|" - 2>/dev/null \
  | utils/int2sym.pl -f 5 data/lang/words.txt \
  | awk -v utt="$utt_id" '$1==utt {print $5, $6}'
