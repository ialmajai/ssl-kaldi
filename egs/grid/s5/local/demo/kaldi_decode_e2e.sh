#!/usr/bin/env bash
# Decode one already-extracted, already-upsampled raw AV-HuBERT utterance
# (base model, layer 4, meanface, 768-dim, 50fps -- upsampled in-process by
# kaldi_lipreader.py, see _pca_and_upsample's sibling _upsample_only; NOT
# PCA'd, unlike kaldi_decode.sh's fMLLR route) through the e2e flat-start
# chain (renorm + no-relu + full CMVN, 8-layer TDNN-F stack, trained on
# clean + one offline-augmented copy of the training set -- flip/rotate/
# crop-jitter/blur/noise on ROI pixels, see augment_roi_frames in
# compute_avhubert_feats.py and NOTES.md's augmentation entries: 6.81%
# in-domain WER, *better* than the clean-only 8-layer model's 6.90%, and
# meaningfully more robust to noise/blur at test time): no GMM, no fMLLR --
# straight nnet3 chain decode. Prints the 1-best transcript to stdout as
# "word conf" lines (raw MBR posterior confidence -- fast, but see
# kaldi_calibrate_conf.sh's docstring for why the number itself is not
# great; that script refines these same words' confidence separately, so
# the transcript can be shown immediately without waiting on it).
#
# Usage: local/demo/kaldi_decode_e2e.sh <utt-id> <feats-dir>
#   <feats-dir> must contain feats.scp/utt2spk/spk2utt for exactly one
#   utterance (768-dim, 50fps, meanface-cropped base-layer-4 raw features).
set -euo pipefail
cd "$(dirname "$0")/../.."  # egs/grid/s5
. ./cmd.sh
. ./path.sh

utt_id=$1
feats_dir=$2

chain=exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x
tree=exp/chain/e2e_tree_lor10_aug2x  # dedicated to this model, not the shared
                                # exp/chain/e2e_tree -- the augmented
                                # training data's utt-ids ("aug-" prefix)
                                # needed their own tree/fst.scp, see
                                # NOTES.md's augmentation training entry

# Per-request decode output lives outside the repo tree by default
# (KALDI_DEMO_SCRATCH unset -> $chain itself, the pre-existing behavior on
# bare metal) so a Docker deployment can mount the repo read-only and point
# this at a writable tmpfs instead (see demo_docker/run.sh) -- matches
# kaldi_calibrate_conf.sh, which reads back from the same location.
scratch=${KALDI_DEMO_SCRATCH:-$chain}
decode_dir=$scratch/decode_demo
mkdir -p "$scratch"

stage() { echo "[stage] $(date +%H:%M:%S.%N | cut -c1-12) $1" >&2; }

stage "cmvn stats start"
steps/compute_cmvn_stats.sh "$feats_dir" >&2
utils/fix_data_dir.sh "$feats_dir" >&2

stage "e2e chain decode start"
# Clear contents, not the directory itself: under Docker, $decode_dir can
# be a tmpfs mount point layered inside an otherwise read-only repo mount
# (see demo_docker/run.sh) -- removing the directory entry itself would
# need write access to its (read-only) parent, even though writing files
# *inside* the tmpfs is fine. Works identically on bare metal, where
# $decode_dir is just a plain directory.
mkdir -p "$decode_dir"
find "$decode_dir" -mindepth 1 -delete
steps/nnet3/decode.sh \
  --acwt 1.0 --post-decode-acwt 10.0 \
  --lattice-beam 2.0 \
  --extra-left-context-initial 0 \
  --extra-right-context-final 0 \
  --frames-per-chunk 150 \
  --nj 1 --cmd "$decode_cmd" --num-threads 4 --skip-scoring true \
  "$tree/graph" "$feats_dir" "$decode_dir" >&2

stage "confidence scoring start"

# Fast path only: raw MBR posterior confidence straight off the lattice
# that decode just produced, no calibration pipeline (that's ~1-2s of
# extra lattice-rescale/depth/logistic-regression work -- see
# kaldi_calibrate_conf.sh, called separately once the text below is
# already on screen). Same underlying lattice either way; this is not a
# lower-quality decode, just an unrefined confidence number.
lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=true \
  "ark:gunzip -c $decode_dir/lat.1.gz|" - 2>/dev/null \
  | utils/int2sym.pl -f 5 data/lang/words.txt \
  | awk -v utt="$utt_id" '$1==utt {print $5, $6}'
