#!/usr/bin/env bash
# Copyright    2017  Hossein Hadian
#              2025  Ibrahim Almajai


# This script performs chain training in a flat-start manner
# and without building or using any context-dependency tree.
# It does not use ivecors or other forms of speaker adaptation.

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
affix=1a

# training options
dropout_schedule='0,0@0.20,0.5@0.50,0'
num_epochs=10
num_jobs_initial=2
num_jobs_final=5
#Suggested:
minibatch_size=150=64,32/200=32,16/500=16,8
common_egs_dir=
l2_regularize=0.00005
frames_per_iter=1000000
cmvn_opts="--norm-means=false --norm-vars=false"
# renorm-linear rescales each input frame to RMS 1.0 before the first affine
# layer; wins or ties relu-batchnorm on 5 of 6 recipes.
input_style=renorm-linear
# tdnnf2..tdnnf(1+N); stride prefix 1,1,1,0 then 2. 10 measured best here
# and on grid/torgo; pass 12 to reproduce the old stack.
num_tdnnf_layers=10
train_set=train_clean_5_spe2e
test_sets="dev_clean_2_raw"
frame_subsampling_factor=2
treedir=exp/chain/e2e_tree  # trivial tree; override for a train set with other utt ids

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

lang=data/lang_nosp_e2e
dir=exp/chain/e2e_tdnnf_${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_nosp $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  echo "$0: Estimating a phone language model for the denominator graph..."
  mkdir -p $treedir/log
  $train_cmd $treedir/log/make_phone_lm.log \
             cat data/$train_set/text \| \
             steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
             data/lang_nosp \| \
             utils/sym2int.pl -f 2- data/lang_nosp/phones.txt \| \
             chain-est-phone-lm --num-extra-lm-states=2000 \
             ark:- $treedir/phone_lm.fst || exit 1;
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj 30 --cmd "$train_cmd" \
                                       --shared-phones true \
                                       data/$train_set $lang $treedir || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  feats_dim=`feat-to-dim scp:data/$train_set/feats.scp  -`
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  if [ "$input_style" = renorm-linear ]; then
    cat <<EOF > $dir/configs/network.xconfig

  input dim=$feats_dim name=input

  renorm-component name=input-renorm input=input target-rms=1.0

  linear-component name=tdnn1 input=Append(-1,0,1) dim=768 orthonormal-constraint=-1.0 l2-regularize=0.01
EOF
  else
    cat <<EOF > $dir/configs/network.xconfig

  input dim=$feats_dim name=input

  relu-batchnorm-dropout-layer name=tdnn1 input=Append(-1,0,1) $tdnn_opts dim=768
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

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=768 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  echo "$0: training the neural net using the chain e2e framework"
  # Only multi-job training needs Exclusive_Process, to stop two jobs landing on
  # the same card. One job runs fine in Default mode.
  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if { [ "$num_jobs_final" -gt 1 ] || [ "$num_jobs_initial" -gt 1 ]; } && \
     [ "$compute_mode" != "Exclusive_Process" ]; then
    echo "$0: more than one training job needs GPU compute mode Exclusive_Process"
    echo "$0: run 'sudo nvidia-smi -c 3', or pass --num-jobs-initial 1 --num-jobs-final 1"
    exit 1
  fi

  echo "$0: training started"
  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.002 \
    --trainer.optimization.final-effective-lrate 0.0002 \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/${train_set} \
    --chain.frame-subsampling-factor ${frame_subsampling_factor} \
    --use-gpu=wait \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 4 ]; then

  utils/lang/check_phones_compatible.sh \
    data/lang_nosp_test_tgsmall/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_nosp_test_tgsmall \
    $dir $treedir/graph_tgsmall || exit 1; 
fi

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    data_affix=$(echo $data | sed s/test_//)
    nspk=$(wc -l <data/${data}/spk2utt)
    
    steps/nnet3/decode.sh \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
      $treedir/graph_tgsmall data/${data} ${dir}/decode_tgsmall_${data_affix} || exit 1
  
    steps/lmrescore.sh --cmd "$decode_cmd" --self-loop-scale 1.0 \
      data/lang_nosp_test_{tgsmall,tgmed} \
      data/${data} ${dir}/decode_{tgsmall,tgmed}_${data_affix}  || exit 1

    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test_{tgsmall,tglarge} \
      data/${data} ${dir}/decode_{tgsmall,tglarge}_${data_affix}  || exit 1     
  done
 
fi

echo "Done. Date: $(date). Training and decoding finished successfully."
