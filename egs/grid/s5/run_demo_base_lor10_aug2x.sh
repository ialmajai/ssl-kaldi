#!/usr/bin/env bash
# Demo model on the LoRA-32 base encoder (input/avhubert_base_ctc_lora.pt) at layer 10:
# the lor10 e2e chain recipe (6.30 / Lombard eval 5.70) trained on clean + one
# augmented copy, as the aug2x demo model was built on frozen L4.
#   ./run_demo_base_lor10_aug2x.sh extract   # local GPU: augmented L10 copy, combined set
#   ./run_demo_base_lor10_aug2x.sh train     # fierce: tree, chain, GRID + Lombard decode
set -uo pipefail
cd /data/learn/ssl-kaldi/egs/grid/s5 || exit 1
export PATH=/data/miniconda3/envs/ssl-kaldi/bin:$PATH
. ./cmd.sh; . ./path.sh
ckpt=input/avhubert_base_ctc_lora.pt
aug=data/train_lor10_raw_aug; comb=data/train_lor10_raw_combined; up=${comb}_upsampled
tree=exp/chain/e2e_tree_lor10_aug2x; affix=lor10norelu_cmvn_aug2x; dir=exp/chain/e2e_tdnnf_$affix
step() { echo "######## $(date +%H:%M:%S) [base-lor10-aug2x] $*"; }

case ${1:-} in
extract)
  if [ ! -s $aug/feats.scp ]; then
    step "extracting augmented copy (seed 42, as aug2x)"
    rm -rf $aug; mkdir -p $aug
    for f in video.scp text utt2spk spk2utt; do cp data/train_l4_raw_aug/$f $aug/; done
    local/make_avhubert_augmented.sh --nj 2 --layer 10 --ckpt $ckpt --augment-seed 42 $aug \
      || { echo "FATAL: augmented extraction"; exit 1; }
    steps/compute_cmvn_stats.sh $aug; utils/fix_data_dir.sh $aug
  fi
  step "aug: $(wc -l < $aug/feats.scp) utts (aug2x had $(wc -l < data/train_l4_raw_aug/feats.scp))"
  rm -rf $comb $up
  utils/combine_data.sh $comb data/train_lor10_raw $aug
  local/copy_data_dir.sh $comb $up
  shared/upsample_features.sh --cmd run.pl --nj 4 --interp-mode nearest --upsample-factor 2 $comb $up \
    || { echo "FATAL: upsample"; exit 1; }
  steps/compute_cmvn_stats.sh $up; utils/fix_data_dir.sh $up
  cp data/train_lor10_raw_upsampled/allowed_{lengths,durs}.txt $up/
  step "combined: $(wc -l < $up/feats.scp) utts"
  echo "DEMO-LOR10-EXTRACT-DONE"
  ;;
train)
  # Stage 1 builds a dedicated tree: the shared trees' FSTs lack the aug- utt-ids.
  step "training e2e chain from stage 1"
  local/chain/e2e/run_tdnn_flatstart.sh --affix $affix --frame-subsampling-factor 2 \
    --cmvn-opts "--norm-means=true --norm-vars=true" --treedir $tree \
    --train-set $(basename $up) --test-set test_lor10_raw_upsampled --stage 1 \
    || { echo "FATAL: training"; exit 1; }
  step "SUMMARY GRID $(grep -h WER $dir/decode/wer_* | utils/best_wer.sh)"
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --lattice-beam 2.0 \
    --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150 \
    --nj 4 --cmd "$decode_cmd" --num-threads 4 \
    $tree/graph data/lor10_lomb_raw_upsampled $dir/decode_lombard || { echo "FATAL: Lombard decode"; exit 1; }
  step "SUMMARY Lombard $(local/score_lombard_split.sh $dir/decode_lombard | tr '\n' ' ')"
  echo "DEMO-LOR10-TRAIN-DONE"
  ;;
*) echo "usage: $0 extract|train"; exit 1;;
esac
