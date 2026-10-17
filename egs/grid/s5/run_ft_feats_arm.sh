#!/usr/bin/env bash
# Features from the fine-tuned AV-HuBERT base encoder (input/avhubert_ft.pt)
# through the frozen base L4 e2e chain recipe (renorm, no relu, CMVN; 7.11),
# so the only change is fine-tuning before extraction. Mean-face crop as the
# frozen arm, although the encoder was fine-tuned on centroid crops.
#   ./run_ft_feats_arm.sh LAYER [CKPT TAG]
# CKPT/TAG default to the fine-tuned encoder; pass input/base_vox_iter5.pt frzl12
# for the frozen control at the same layer. Run from egs/grid/s5 in the
# ssl-kaldi environment, after run.sh (data/{train,test}_l4_raw).
set -uo pipefail
. ./cmd.sh; . ./path.sh
compress=${COMPRESS:-true}   # COMPRESS=false keeps float32 features
featdim=${FEATDIM:-768}      # FEATDIM=1024 for AV-HuBERT large
layer=$1; ckpt=${2:-input/avhubert_ft.pt}; tag=${3:-ftl$layer}
seed=${SEED:-0}             # SEED=1,2 retrain the chain with another --srand
cmvn=${CMVN_OPTS:-"--norm-means=true --norm-vars=true"}
affix=${AFFIX:-${tag}norelu_cmvn$([ $seed = 0 ] || echo _s$seed)}
store=${STORE:-$PWD/data/ft_store}/ssl-kaldi-grid-$tag   # STORE=/big/disk puts the archives elsewhere
step() { echo "######## $(date +%H:%M:%S) [$tag] $*"; }
# Claim the arm so this host and fierce never train the same one; the other skips.
claim=exp/chain/.claim_$tag
if [ -s $claim ] && [ "$(cat $claim)" != "$(hostname)" ]; then
  step "claimed by $(cat $claim), skipping"; exit 0
fi
hostname > $claim

for x in train test; do
  raw=data/${x}_${tag}_raw; up=${raw}_upsampled
  if [ ! -s $up/feats.scp ]; then
    step "extracting $x"
    rm -rf $raw $up $store/$x; mkdir -p $store/$x/raw $store/$x/up
    local/copy_data_dir.sh data/${x}_l4_raw $raw; rm -f $raw/feats.scp $raw/cmvn.scp
    ln -sfn $store/$x/raw $raw/data
    local/make_avhubert.sh --cmd run.pl --compress $compress --nj 2 --ckpt $ckpt \
      --avhubert-path $PWD/av_hubert --layer $layer --crop-mode meanface $raw \
      || { echo "FATAL: extraction $x"; exit 1; }
    steps/compute_cmvn_stats.sh $raw; utils/fix_data_dir.sh $raw
    local/copy_data_dir.sh $raw $up; ln -sfn $store/$x/up $up/data
    shared/upsample_features.sh --cmd run.pl --compress $compress --nj 4 --interp-mode nearest \
      --upsample-factor 2 $raw $up || { echo "FATAL: upsample $x"; exit 1; }
    steps/compute_cmvn_stats.sh $up; utils/fix_data_dir.sh $up
    step "$x: $(wc -l < $up/feats.scp) utts (frozen arm: $(wc -l < data/${x}_l4_raw/feats.scp))"
  fi
done

# The shared exp/chain/e2e_tree lost its fst.*.scp and 0.mdl in a cleanup, so
# rebuild it (stage 1) into its own dir; same text and lang give the same tree.
tree=exp/chain/e2e_tree_ft
start=2; [ -s $tree/fst.1.scp ] || start=1
# GRID clips are 3 s; run_e2e.sh fixes these for the 50 fps upsampled features.
echo -e '2.96\n3.00' > data/train_${tag}_raw_upsampled/allowed_durs.txt
echo -e '148\n150' > data/train_${tag}_raw_upsampled/allowed_lengths.txt
if [ $start = 1 ]; then
  # Build $tree first: stage 1 of run_tdnn_flatstart.sh with --treedir $tree, then
  # copy exp/chain/e2e_tdnnf_l4norelu_cmvn/phone_lm.fst over $tree/phone_lm.fst so
  # the denominator graph matches the frozen arm (the rebuilt phone LM differs).
  echo "FATAL: $tree missing; build it as above"; exit 1
fi
step "training e2e chain from stage $start"
local/chain/e2e/run_tdnn_flatstart.sh --affix $affix --srand $seed --frame-subsampling-factor 2 --feat-dim $featdim \
  --cmvn-opts "$cmvn" --treedir $tree \
  --train-set train_${tag}_raw_upsampled --test-set test_${tag}_raw_upsampled --stage $start \
  || { echo "FATAL: training"; exit 1; }
step "SUMMARY $(grep -h WER exp/chain/e2e_tdnnf_$affix/decode*/wer_* | utils/best_wer.sh)"
echo "GRID-$tag-DONE"
