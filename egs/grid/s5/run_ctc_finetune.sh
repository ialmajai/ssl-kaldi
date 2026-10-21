#!/usr/bin/env bash
# Copyright   2026  Ibrahim Almajai
# License: Apache 2.0.
#
# CTC fine-tuning of AV-HuBERT on GRID (LoRA or matched full fine-tune), then
# the fine-tuned encoder as a feature extractor for the flat-start e2e chain.
# Run after run.sh stages 0-2 (data/{train,test}_raw).
#
#   ./run_ctc_finetune.sh                                   # LoRA r32, base, layer 10
#   ./run_ctc_finetune.sh --config base_ctc_full            # matched full fine-tune
#   ./run_ctc_finetune.sh --config large_ctc_lora --layer 20
#
# Configs live in local/finetune/conf: base_ctc_lora, base_ctc_full,
# large_ctc_lora, large_ctc_full. Every config shares one recipe: letter CTC
# head, encoder frozen for the first 3,000 of 15,000 updates, 96 px mean-face
# ROIs with random 88 px crops, validation on held-out training speakers s7 and
# s30, ResNet BatchNorm statistics frozen.
#
# Stages:
#   0  96 px mean-face ROIs for train and test
#   1  fairseq manifests (letters; s7, s30 held out for validation)
#   2  fine-tune (one GPU; restartable, fairseq resumes from checkpoint_last.pt)
#   3  merge LoRA into a plain AV-HuBERT checkpoint: input/avhubert_<config>.pt
#   4  the fine-tuned model decoding itself (greedy CTC, no LM) on test
#   5  features at --layer for train and test, x2 upsampled
#   6  flat-start e2e chain (renorm, no relu, full CMVN), decoded on test

set -euo pipefail

stage=0
stop_stage=100
config=base_ctc_lora
pretrained=        # empty: input/{base,large}_vox_iter5.pt from the config name
layer=             # empty: 10 for base, 20 for large
ft_dir=finetune_ctc
nj_roi=8
nj_feats=2
avhubert_path=$PWD/av_hubert

# Take --config before utils/parse_options.sh, which sources any --config as a Kaldi config file.
args=()
while [ $# -gt 0 ]; do
  if [ "$1" = --config ]; then config=$2; shift 2; else args+=("$1"); shift; fi
done
set -- ${args[@]+"${args[@]}"}

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

case $config in
  base_*)  : ${pretrained:=input/base_vox_iter5.pt}  ${layer:=10} ;;
  large_*) : ${pretrained:=input/large_vox_iter5.pt} ${layer:=20} ;;
  *) echo "$0: --config must start with base_ or large_, got '$config'"; exit 1 ;;
esac
[ -f local/finetune/conf/$config.yaml ] || { echo "$0: no local/finetune/conf/$config.yaml"; exit 1; }
[ -f $pretrained ] || { echo "$0: $pretrained missing (run.sh stage 2 downloads it)"; exit 1; }

export PYTHONPATH=$avhubert_path/fairseq:$avhubert_path${PYTHONPATH:+:$PYTHONPATH}
export AVHUBERT_ROOT=$avhubert_path
exp=$ft_dir/exp_$config
ckpt=input/avhubert_$config.pt
tag=${config}_L$layer
chain=exp/chain/e2e_tdnnf_${tag}
tree=exp/chain/e2e_tree_ft

run() { [ $stage -le $1 ] && [ $stop_stage -ge $1 ]; }

if run 0; then
  for x in train test; do
    src=data/${x}_raw
    [ -f $src/video.scp ] || { echo "$0: $src/video.scp missing"; exit 1; }
    echo "$0: mouth ROIs for $x"
    OMP_NUM_THREADS=1 python local/finetune/dump_roi_meanface.py $src/video.scp \
      $ft_dir/rois96_$x --nj $nj_roi --path $avhubert_path \
      --mean-face $avhubert_path/avhubert/20words_mean_face.npy
  done
fi

if run 1; then
  python local/finetune/prepare_ctc_manifests.py --data data/train_raw \
    --roi-dir $ft_dir/rois96_train --out $ft_dir/manifest
fi

if run 2; then
  python -m fairseq_cli.hydra_train --config-dir $PWD/local/finetune/conf --config-name $config \
    hydra.run.dir=$PWD/$exp common.user_dir=$PWD/local/finetune/avh_lora \
    common.tensorboard_logdir=$PWD/$exp/tb \
    task.data=$PWD/$ft_dir/manifest task.label_dir=$PWD/$ft_dir/manifest \
    model.w2v_path=$PWD/$pretrained checkpoint.save_dir=$PWD/$exp/checkpoints
fi

if run 3; then
  python local/finetune/export_lora_encoder.py $exp/checkpoints/checkpoint_best.pt $pretrained $ckpt
fi

if run 4; then
  for x in test; do
    src=data/${x}_raw
    [ -d $ft_dir/rois96_$x ] || continue
    hyp=$exp/decode_ctc_$x.txt
    # Without the training checkpoint, rebuild the model from the merged encoder
    # and the CTC head that stage 3 saved beside it.
    model=($exp/checkpoints/checkpoint_best.pt)
    [ -f ${model[0]} ] || model=($ckpt --head ${ckpt%.pt}.ctc_head.pt)
    python local/finetune/decode_ctc.py ${model[0]} $ft_dir/rois96_$x \
      $src $hyp --manifest-dir $exp/manifest_$x --label-dict $ft_dir/manifest/dict.ltr.txt \
      "${model[@]:1}"
    compute-wer --text --mode=present ark:$src/text ark:$hyp | grep WER
  done
fi

if run 5; then
  for x in train test; do
    src=data/${x}_raw
    raw=data/${x}_${tag}_raw; up=${raw}_upsampled
    min=74
    rm -rf $raw $up
    local/copy_data_dir.sh $src $raw; rm -f $raw/feats.scp $raw/cmvn.scp
    local/make_avhubert.sh --cmd "$feats_cmd" --nj $nj_feats --ckpt $ckpt --min-frames $min \
      --avhubert-path $avhubert_path --layer $layer --crop-mode meanface $raw
    steps/compute_cmvn_stats.sh $raw; utils/fix_data_dir.sh $raw
    local/copy_data_dir.sh $raw $up
    shared/upsample_features.sh --cmd "$decode_cmd" --nj 4 --interp-mode nearest \
      --upsample-factor 2 $raw $up
    steps/compute_cmvn_stats.sh $up; utils/fix_data_dir.sh $up
  done
fi

if run 6; then
  # GRID clips are 3 s: 148-150 frames after x2 upsampling.
  printf '2.96\n3.00\n' > data/train_${tag}_raw_upsampled/allowed_durs.txt
  printf '148\n150\n' > data/train_${tag}_raw_upsampled/allowed_lengths.txt
  # Every fine-tuned arm shares one tree; build it on the first run.
  start=2; [ -s $tree/fst.1.scp ] || start=0
  local/chain/e2e/run_tdnn_flatstart.sh --affix $tag --frame-subsampling-factor 2 \
    --cmvn-opts "--norm-means=true --norm-vars=true" --treedir $tree \
    --train-set train_${tag}_raw_upsampled --test-set test_${tag}_raw_upsampled --stage $start
  echo "$0: GRID test $(grep -h WER $chain/decode/wer_* | utils/best_wer.sh)"
fi
