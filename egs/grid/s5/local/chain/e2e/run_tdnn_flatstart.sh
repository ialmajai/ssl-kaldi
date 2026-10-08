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
num_jobs_initial=1
num_jobs_final=1
#Suggested:
minibatch_size=150=64,32/200=32,16/500=16,8
common_egs_dir=
l2_regularize=0.00005
frames_per_iter=500000
cmvn_opts="--norm-means=false --norm-vars=false"
train_set=train_raw
test_set=test_raw
frame_subsampling_factor=2
num_tdnnf_layers=12  # tdnnf2..tdnnf(1+N); stride pattern is the fixed
                      # 1,1,1,0 prefix (truncated if N<4) then stride=2 for
                      # the rest, matching the existing 12-layer scheme
feat_dim=768
treedir=exp/chain/e2e_tree  # shared by every e2e model in this recipe --
                             # only override for a train_set with a
                             # different utt-id set (e.g. an augmented
                             # copy added under new utt-ids), since the
                             # tree's fst.scp/phone-LM are built from one
                             # specific train_set's utt-ids/text at stage 1


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

lang=data/lang_e2e
dir=exp/chain/e2e_tdnnf_${affix}


if [ $stage -le 0 ]; then
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

if [ $stage -le 1 ]; then
  echo "$0: Estimating a phone language model for the denominator graph..."
  mkdir -p $treedir/log
  $train_cmd $treedir/log/make_phone_lm.log \
             cat data/$train_set/text \| \
             steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.0 \
             data/lang \| \
             utils/sym2int.pl -f 2- data/lang/phones.txt \| \
             chain-est-phone-lm --num-extra-lm-states=2000 \
             ark:- $treedir/phone_lm.fst
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj 20 --cmd "$train_cmd" \
                                       --shared-phones false \
				       --min-biphone-count 50 \
				       --min-monophone-count 10 \
				       data/$train_set $lang $treedir
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=$feat_dim name=input

  renorm-component name=input-renorm input=input target-rms=1.0

  # no relu here: AV-HuBERT's own nonlinearity is GELU, and its hidden states
  # are roughly zero-centered in both directions (unlike log-energy acoustic
  # features) -- a hard relu on the very first layer, before the network has
  # adapted, discards about half that signal by construction. Pure linear
  # reprojection instead, matching the projection variant's ssl_proj shape;
  # nonlinear processing happens downstream in the (unchanged) TDNN-F stack.
  linear-component name=tdnn1 input=Append(-1,0,1) dim=768 orthonormal-constraint=-1.0 l2-regularize=0.01
EOF
  stride_prefix=(1 1 1 0)
  for i in $(seq 1 $num_tdnnf_layers); do
    if [ $i -le 4 ]; then
      stride=${stride_prefix[$((i-1))]}
    else
      stride=2
    fi
    echo "  tdnnf-layer name=tdnnf$((i+1)) $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=$stride" >> $dir/configs/network.xconfig
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
  # With a single training job (num_jobs_final=1) one GPU process suffices, so
  # Default compute mode is fine; only warn if multiple jobs need exclusive access.
  if [ "$num_jobs_final" -gt 1 ] || [ "$num_jobs_initial" -gt 1 ]; then
    compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
    if [ "$compute_mode" != "Exclusive_Process" ]; then
      echo "num_jobs_final>1 needs GPU compute mode Exclusive_Process (run: sudo nvidia-smi -c 3),"
      echo "or set --num-jobs-final 1 to train with a single GPU job in Default mode."
      exit 1
    fi
  fi

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
    --trainer.optimization.initial-effective-lrate 0.004 \
    --trainer.optimization.final-effective-lrate 0.0004 \
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
    data/lang/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang \
    $dir $treedir/graph || exit 1; 
fi

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

      nspk=$(wc -l <data/$test_set/spk2utt)
      steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --lattice-beam 2.0 \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        $treedir/graph data/$test_set ${dir}/decode || exit 1    
 
fi

echo "Done. Date: $(date). Training and decoding finished successfully."
