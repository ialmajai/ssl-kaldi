#!/usr/bin/env bash

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

stage=15
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
feat_suffix=  # pick an alternative feature set: data/<set>_sp<suffix>_raw and
              # data/<test><suffix>_raw, while tree/alignments/lattices stay
              # those of the plain train_set (shared supervision, so feature
              # A/Bs are not confounded by alignment quality)
gmm=tri3b
nnet3_affix=_online_cmn
online_cmvn=true
affix=
tree_affix=
# renorm-linear rescales each input frame to RMS 1.0 and drops the first-layer
# relu; beat relu-batchnorm here 3.88 vs 4.42 tglarge in a controlled A/B
# (both with online CMVN, 12L, jobs 1/1 -- FINDINGS.md sec 11).
input_style=renorm-linear
# tdnnf2..tdnnf(1+N). Stride prefix 1,1,1,0 then 2, so 12 keeps the historical
# stack exactly and 10 trims two stride-2 layers off the top.
num_tdnnf_layers=12
# Keep online CMVN with renorm-linear: the controlled A/B kept it on both
# arms and renorm still won, refuting the earlier redundancy guess.
cmvn_opts="--config=conf/online_cmvn.conf"
# 2/5 is the historical setting and needs a card per job. TIMIT measured 1/1 as
# better as well as cheaper, since Kaldi scales the effective learning rate by
# job count. Kept at 2/5 here so existing runs reproduce.
num_jobs_initial=2
num_jobs_final=5
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
common_egs_dir=
xent_regularize=0.1
frame_subsampling_factor=2

# training options
srand=0
remove_egs=true
reporting_email=

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

case $input_style in
  renorm-linear|relu-batchnorm) ;;
  *) echo "$0: --input-style must be renorm-linear or relu-batchnorm, got '$input_style'"; exit 1 ;;
esac

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}_subsamp_${frame_subsampling_factor}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp${feat_suffix}_raw
lores_train_data_dir=data/${train_set}_sp_pca

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 15 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  
fi

if [ $stage -le 16 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 17 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
 
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${frame_subsampling_factor} \
    --context-opts "--context-width=1 --central-position=0" \
    --cmd "$train_cmd" 2000 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

if [ $stage -le 18 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  feats_dim=`feat-to-dim scp:$train_data_dir/feats.scp  -`
  tdnn_opts="l2-regularize=0.03"
  tdnnf_opts="l2-regularize=0.03 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

  mkdir -p $dir/configs
  if [ "$input_style" = renorm-linear ]; then
    # No relu on the first layer. HuBERT's own nonlinearity is GELU and its
    # hidden states are roughly zero-centered, unlike log-energy features, so a
    # hard relu here discards about half the signal by construction.
    cat <<EOF > $dir/configs/network.xconfig
  input dim=$feats_dim name=input

  renorm-component name=input-renorm input=input target-rms=1.0

  linear-component name=tdnn1 input=Append(-1,0,1) dim=768 orthonormal-constraint=-1.0 $tdnn_opts
EOF
  else
    cat <<EOF > $dir/configs/network.xconfig
  input dim=$feats_dim name=input

  no-op-component name=input2 input=Append(-1,0,1)

  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=768 input=input2
EOF
  fi
  stride_prefix=(1 1 1 0)
  for i in $(seq 1 $num_tdnnf_layers); do
    if [ $i -le 4 ]; then stride=${stride_prefix[$((i-1))]}; else stride=2; fi
    echo "  tdnnf-layer name=tdnnf$((i+1)) $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=$stride" \
      >> $dir/configs/network.xconfig
  done
  cat <<EOF >> $dir/configs/network.xconfig

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

if [ $stage -le 19 ]; then
  # Only multi-job training needs Exclusive_Process: concurrent
  # nnet3-chain-train processes reserve about 4.6 GB each and will not share a
  # single card in Default mode. One job is fine in either mode.
  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if { [ "$num_jobs_final" -gt 1 ] || [ "$num_jobs_initial" -gt 1 ]; } && [ "$compute_mode" != "Exclusive_Process" ]; then
    echo "$0: more than one training job needs GPU compute mode Exclusive_Process"
    echo "$0: run 'sudo nvidia-smi -c 3', or pass --num-jobs-initial 1 --num-jobs-final 1"
    exit 1
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=1000000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --chain.frame-subsampling-factor ${frame_subsampling_factor} \
    --chain.alignment-subsampling-factor ${frame_subsampling_factor} \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0  --online-cmvn $online_cmvn" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 20 ]; then
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 21 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for dataset in $test_sets; do
    
      nspk=$(wc -l <data/${dataset}${feat_suffix}_raw/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $tree_dir/graph_tgsmall data/${dataset}${feat_suffix}_raw ${dir}/decode_tgsmall_${dataset} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
      data/${dataset}${feat_suffix}_raw ${dir}/decode_{tgsmall,tglarge}_${dataset} || exit 1
    
  done
fi

exit 0;
