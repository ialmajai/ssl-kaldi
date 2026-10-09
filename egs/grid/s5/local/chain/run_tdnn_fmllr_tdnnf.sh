#!/bin/bash
set -e

# Train a TDNN-F chain on fMLLR features from the GMM-SAT system (run.sh stage 13).
# Needs data/$train_set (the GMM's features), $data_fmllr/{$train_set,$test_set}
# and ${gmm}_ali, all written by run.sh stages 4-12.

# configs for 'chain'
stage=9
train_stage=-10
get_egs_stage=-10
speed_perturb=false
affix=tdnnf
suffix=
frame_subsampling_factor=2
decode_iter=
decode_nj=4
# training options
minibatch_size=64
frames_per_eg=150
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dim=600
small_dim=256
num_jobs_initial=1
num_jobs_final=1
train_set=train
test_set=test
gmm=exp/tri3b
data_fmllr=data-fmllr-tri3b

# End configuration section.
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

dir=exp/chain/tdnn${affix}${suffix}
ali_dir=${gmm}_ali$suffix
treedir=exp/chain/tri5_2000_tree$suffix
lang=data/lang_chain_2y

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang $gmm exp/tri3b_lats$suffix
  rm exp/tri3b_lats$suffix/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor ${frame_subsampling_factor} \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 2000 data/$train_set $lang $ali_dir $treedir || exit 1
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnnf_opts="l2-regularize=0.004 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.004"
  prefinal_opts="l2-regularize=0.004"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 l2-regularize=0.004 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true dim=$dim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$dim bottleneck-dim=$small_dim time-stride=2
  linear-component name=prefinal-l dim=$small_dim $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$dim small-dim=$small_dim
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$dim small-dim=$small_dim
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1000000 \
    --trainer.num-epochs 10 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --chain.frame-subsampling-factor ${frame_subsampling_factor} \
    --chain.alignment-subsampling-factor ${frame_subsampling_factor} \
    --trainer.max-param-change 2.0 \
    --trainer.optimization.max-models-combine 8 \
    --cleanup.remove-egs $remove_egs \
    --use-gpu=wait \
    --feat-dir ${data_fmllr}/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_lats$suffix \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 15 ]; then
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 4 --cmd "$decode_cmd"  \
      $graph_dir ${data_fmllr}/${test_set} \
      $dir/decode_test || exit 1;
fi

exit 0;
