#!/usr/bin/env bash
# Stages 4-5 of run_ctc_finetune.sh --arm full, with the features and the
# chain dir optionally on another disk: layer 9 of the fully fine-tuned
# mHuBERT-147 through the same renorm-linear n10 e2e chain as the frozen
# 19.47 and the LoRA 19.07.
#   ./run_ft_full_chain.sh [--store /big/disk/swahili-ft-full]
set -uo pipefail
. ./cmd.sh; . ./path.sh

model=exp_ft/mhubert147_full
layer=9
store=   # where features and the chain dir go; default data/ft_full_store
. utils/parse_options.sh
[ -z "$store" ] && store=$PWD/data/ft_full_store
mkdir -p $store

for pair in "train_spe2e train_spe2e_ft_full" "test test_ft_full"; do
  set -- $pair
  if [ ! -s data/$2/feats.scp ]; then
    echo "######## $(date +%H:%M:%S) extracting $model L$layer -> $2"
    rm -rf $store/$2 data/$2
    utils/copy_data_dir.sh data/$1 $store/$2 || exit 1
    rm -f $store/$2/feats.scp $store/$2/cmvn.scp
    for f in allowed_lengths.txt allowed_durs.txt frame_shift; do
      [ -f data/$1/$f ] && cp data/$1/$f $store/$2/
    done
    ln -sfn $store/$2 data/$2
    shared/make_ssl.sh --cmd "$train_cmd" --nj 4 --ssl-model "$model" \
      --layer $layer data/$2 || { echo "EXTRACTION FAILED $2"; exit 1; }
    steps/compute_cmvn_stats.sh data/$2 || exit 1
  fi
done

# Keep the chain dir (egs, models) on fierce's disk too.
dir=exp/chain/e2e_tdnnf_n10_ft_full
if [ ! -L $dir ]; then
  rm -rf $dir; mkdir -p $store/e2e_tdnnf_n10_ft_full; ln -sfn $store/e2e_tdnnf_n10_ft_full $dir
fi

echo "######## $(date +%H:%M:%S) training e2e n10 on fine-tuned (full) mHuBERT-147 L9"
rm -f exp/chain/e2e_tree/graph/HCLG.fst  # force rebuild for this model
bash local/chain/e2e/run_tdnn_flatstart.sh --affix n10_ft_full \
  --train-set train_spe2e_ft_full --test-set test_ft_full \
  --input-style renorm-linear --num-tdnnf-layers 10 \
  --num-jobs-initial 1 --num-jobs-final 1 \
  --stage 2 || { echo "FT-FULL n10 FAILED"; exit 1; }

for d in $dir/decode_*; do
  echo "######## $(date +%H:%M:%S) SUMMARY $(basename $d) $(grep -h WER $d/wer_* | utils/best_wer.sh)"
done
echo "FT-FULL-CHAIN-DONE"
