#!/usr/bin/env bash
# Copyright  2026    Ibrahim Almajai
# Apache 2.0

# Chain (LF-MMI) TDNN-F trained directly on the raw SSL features.
# Run local/chain/run_common_ssl.sh first.

set -euo pipefail

stage=0
decode_nj=8
train_set=train
test_sets="dev test"
gmm=tri3
nnet3_affix=
affix=1a
tree_affix=
train_stage=-10
get_egs_stage=-10

num_leaves=2000
# Job schedule for chain training; set both to 1 when training on a single GPU.
num_jobs_initial=2
num_jobs_final=4
# SSL features are already at a 20 ms frame shift, so a subsampling factor of 2
# gives 40 ms output frames. Try 1 if you want output at the feature rate.
frame_subsampling_factor=2
chunk_width=140,100,160
common_egs_dir=
xent_regularize=0.1
srand=0
remove_egs=true
reporting_email=

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
# The subsampling factor is part of the tree directory name: a tree built for
# one factor is not valid for another, and they must not share a directory.
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}_subsamp_${frame_subsampling_factor}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp
train_data_dir=data/${train_set}_sp_raw
lores_train_data_dir=data/${train_set}_sp_pca

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  # $lang is a copy of data/lang with a chain topology, so it is cheap and safe
  # to rebuild whenever it is older than its source. Anything that re-runs
  # prepare_lang (run-mfcc.sh, for instance) restamps data/lang and would
  # otherwise leave this stage permanently stuck.
  if [ -d $lang ] && [ $lang/L.fst -nt data/lang/L.fst ]; then
    echo "$0: $lang is newer than data/lang; keeping it"
  else
    echo "$0: (re)creating lang directory $lang with chain-type topology"
    rm -rf $lang
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  nj=$(cat $ali_dir/num_jobs)
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 2 ]; then
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${frame_subsampling_factor} \
    --cmd "$train_cmd" $num_leaves ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  feats_dim=$(feat-to-dim scp:$train_data_dir/feats.scp -)

  tdnn_opts="l2-regularize=0.03"
  tdnnf_opts="l2-regularize=0.03 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feats_dim name=input

  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=768 input=Append(-1,0,1)

  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=2
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=2
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=2
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=2

  linear-component name=prefinal-l dim=192 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 4 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=8 \
    --trainer.frames-per-iter=1000000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --chain.frame-subsampling-factor ${frame_subsampling_factor} \
    --chain.alignment-subsampling-factor ${frame_subsampling_factor} \
    --egs.chunk-width=$chunk_width \
    --egs.stage=$get_egs_stage \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_bg \
    $tree_dir $tree_dir/graph_bg || exit 1;
fi

if [ $stage -le 6 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  for x in $test_sets; do
    steps/nnet3/decode.sh \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $decode_nj --cmd "$decode_cmd" --num-threads 4 \
      $tree_dir/graph_bg data/${x}_raw ${dir}/decode_$x || exit 1
  done
fi

exit 0
