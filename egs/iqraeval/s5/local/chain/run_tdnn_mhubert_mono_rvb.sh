#!/usr/bin/env bash
set -euo pipefail

stage=0
decode_nj=10
train_set=train
test_sets=dev
gmm=tri2
nnet3_affix=

exp=exp
num_data_reps=1
nj=50

affix=_1a   # affix for the TDNN directory name
tree_affix=mono
train_stage=-10
get_egs_stage=-10
decode_iter=

chunk_width=140,100,160
frame_subsampling_factor=2
common_egs_dir=
xent_regularize=0.1
srand=0
remove_egs=true
reporting_email=

echo "$0 $@"  # Print command line
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

gmm_dir=$exp/$gmm
ali_dir=$exp/${gmm}_ali_${train_set}_sp
tree_dir=$exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
clean_lat_dir=$exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=$exp/chain${nnet3_affix}/tdnn${affix}_sp${tree_affix:+_$tree_affix}_rvb${num_data_reps}
train_data_dir=data/${train_set}_sp_raw_rvb${num_data_reps}

lores_train_data_dir=data/${train_set}_sp_pca
lat_dir=${clean_lat_dir}_rvb${num_data_reps}

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 0 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $clean_lat_dir
  rm $clean_lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 2 ]; then
  clean_lat_nj=$(cat $clean_lat_dir/num_jobs)
  mkdir -p $lat_dir/temp/

  $train_cmd --max-jobs-run 10 JOB=1:$clean_lat_nj \
    $lat_dir/temp/log/copy_clean_lats.JOB.log lattice-copy "ark:gunzip -c \
    $clean_lat_dir/lat.JOB.gz |" ark,scp:$lat_dir/temp/lats.JOB.ark,$lat_dir/temp/lats.JOB.scp

  rm -f $lat_dir/temp/combined_lats.scp

  for i in `seq 0 $num_data_reps`; do
    for n in $(seq $clean_lat_nj); do
      cat $lat_dir/temp/lats.$n.scp
    done | sed -e "s/^/rev${i}-/"
  done >> $lat_dir/temp/combined_lats.scp

  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  utils/split_data.sh $train_data_dir $nj

  $train_cmd --max-jobs-run 10 JOB=1:$nj $lat_dir/copy_combined_lats.JOB.log \
    lattice-copy --include=$train_data_dir/split$nj/JOB/utt2spk \
    scp:$lat_dir/temp/combined_lats_sorted.scp \
    "ark:|gzip -c >$lat_dir/lat.JOB.gz" || exit 1;

  echo $nj > $lat_dir/num_jobs
  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $clean_lat_dir/$f $lat_dir/$f
  done
fi

if [ $stage -le 2 ]; then
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${frame_subsampling_factor} \
    --context-opts "--context-width=1 --central-position=0" \
    --cmd "$train_cmd" 2000 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  tdnn_opts="l2-regularize=0.03"
  tdnnf_opts="l2-regularize=0.03 bypass-scale=0.66"
  linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.015"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=768 name=input

  relu-batchnorm-layer name=tdnn1 dim=768 input=Append(-1,0,1)

  tdnnf-layer name=tdnnf1 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=2

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
    --trainer.num-epochs=5 \
    --trainer.frames-per-iter=1000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.num-chunk-per-minibatch=256,128,64 \
    --chain.frame-subsampling-factor ${frame_subsampling_factor} \
    --chain.alignment-subsampling-factor ${frame_subsampling_factor} \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--max-shuffle-jobs-run 6" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_trigram \
    $tree_dir $tree_dir/graph_tg || exit 1;
fi

if [ $stage -le 6 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for dataset in $test_sets; do    
      nspk=$(wc -l <data/${dataset}/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj 20 --cmd "$decode_cmd"  --num-threads 4 \
          $tree_dir/graph_tg data/${dataset}_raw ${dir}/decode_${dataset}_tg || exit 1    
  done
fi

exit 0 
