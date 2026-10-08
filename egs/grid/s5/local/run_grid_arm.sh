#!/usr/bin/env bash
# Copyright 2026  Ibrahim Almajai
# Build ONE GRID acoustic-model arm (extract standard GRID with a given crop mode
# + AV-HuBERT model, PCA, upsample, mono->tri3b, decode test) in a parallel
# `${aff}` namespace. Used to make crop-vs-crop A/B arms that share the SAME
# encoder, so the comparison isolates the crop method (see the base-for-both
# decision: centroid-base vs meanface-base).
#
# GPU-gated so it never competes with other jobs.

set -euo pipefail
stage=0
free_mib=7000
crop_mode=meanface                 # meanface | centroid (legacy)
aff=_mf                            # namespace suffix, e.g. _mf (meanface), _cb (centroid-base)
avhubert_model=base_vox_iter5.pt   # base -> 768-dim
encoder_layer=9
min_frames=74
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

njfeats=8; pca_dim=30; pca_train_utts=28000; njpca=15; interp_mode=nearest; upsample_factor=2
nj=8; njtest=4
numLeavesTri1=2000;  numGaussTri1=10000
numLeavesMLLT=3500;  numGaussMLLT=20000
numLeavesSAT=5000;   numGaussSAT=40000
avhubert_path=$PWD/av_hubert

if [ $stage -le 0 ]; then
  echo "$(date '+%H:%M:%S') waiting for >=${free_mib} MiB free on the GPU..."
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "${free:-0}" -ge "$free_mib" ] && { echo "GPU free (${free} MiB)"; break; }
    sleep 60
  done
fi

if [ $stage -le 1 ]; then
  for x in train test; do
    rm -rf data/${x}${aff}_raw
    local/copy_data_dir.sh data/$x data/${x}${aff}_raw
    local/make_avhubert.sh --cmd "$feats_cmd" --nj $njfeats --ckpt input/$avhubert_model \
      --avhubert-path $avhubert_path --layer $encoder_layer \
      --crop-mode $crop_mode --min-frames $min_frames data/${x}${aff}_raw
    steps/compute_cmvn_stats.sh data/${x}${aff}_raw
    utils/fix_data_dir.sh data/${x}${aff}_raw
  done
fi

if [ $stage -le 2 ]; then
  pca_dir=pca; mkdir -p $pca_dir; pca_model=pca-${pca_dim}d${aff}.pt
  python shared/pca.py --pca_dim=$pca_dim --mode=train \
    --feats_scp=data/train${aff}_raw/feats.scp --pca_model=$pca_dir/$pca_model \
    --max_utts=$pca_train_utts
  for x in train test; do
    local/copy_data_dir.sh data/${x}${aff}_raw data/${x}${aff}
    shared/make_pca_features.sh --cmd "$decode_cmd" --nj $njpca \
      --pca-model $pca_dir/$pca_model data/${x}${aff}_raw data/${x}${aff}
    steps/compute_cmvn_stats.sh data/${x}${aff}; utils/fix_data_dir.sh data/${x}${aff}
  done
fi

if [ $stage -le 3 ]; then
  for x in train test; do
    local/copy_data_dir.sh data/${x}${aff} data/${x}${aff}_up
    shared/upsample_features.sh --cmd "$decode_cmd" --nj $njpca \
      --interp-mode $interp_mode --upsample-factor $upsample_factor data/${x}${aff} data/${x}${aff}_up
    steps/compute_cmvn_stats.sh data/${x}${aff}_up; utils/fix_data_dir.sh data/${x}${aff}_up
  done
fi

tr=data/train${aff}_up; te=data/test${aff}_up; E=exp${aff}
if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" --boost-silence 1.5 $tr data/lang $E/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" $tr data/lang $E/mono $E/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.5 \
    $numLeavesTri1 $numGaussTri1 $tr data/lang $E/mono_ali $E/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" $tr data/lang $E/tri1 $E/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT $numGaussMLLT \
    $tr data/lang $E/tri1_ali $E/tri2b
  steps/align_si.sh --nj $nj --cmd "$train_cmd" $tr data/lang $E/tri2b $E/tri2b_ali
  steps/train_sat.sh --cmd "$train_cmd" $numLeavesSAT $numGaussSAT \
    $tr data/lang $E/tri2b_ali $E/tri3b
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh data/lang $E/tri3b $E/tri3b/graph
  steps/decode_fmllr.sh --nj $njtest --cmd "$decode_cmd" $E/tri3b/graph $te $E/tri3b/decode
fi

echo -n "arm ${aff} (${crop_mode}, ${avhubert_model}) tri3b: "
grep -h WER $E/tri3b/decode/wer_* 2>/dev/null | utils/best_wer.sh || echo none
echo "GRID_ARM_DONE${aff} $(date '+%H:%M:%S')"
