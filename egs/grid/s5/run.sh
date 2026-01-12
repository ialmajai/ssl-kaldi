#!/bin/bash
# Copyright   2025  Ibrahim Almajai
# License: Apache 2.0.

set -e

stage=0
nj=8
njtest=4
njfeats=10

pca_dim=30
pca_train_utts=28000
interp_mode=nearest
upsample_factor=2

#path to grid corpus dataset
datasrc=grid-corpus

# avhubert_ckpt=/data/git/av_hubert/avhubert/checkpoints/lrs3_vox/clean-pretrain/large_vox_iter5.pt
# encoder_layer=12
avhubert_ckpt=/data/git/av_hubert/avhubert/checkpoints/lrs3_vox/clean-pretrain/base_vox_iter5.pt
encoder_layer=9


#path to avhubert codebase
avhubert_path=/data/git/ssl-kaldi/egs/grid/s5/av_hubert

lang=data/lang
dict=data/local/dict
langtmp=data/local/lang
mkdir -p $langtmp
mkdir -p $dict

monogauss=1000
numLeavesTri1=2000
numGaussTri1=10000
numLeavesMLLT=3500
numGaussMLLT=20000
numLeavesSAT=5000
numGaussSAT=40000

. cmd.sh
. path.sh
. ./utils/parse_options.sh

if [ $stage -le -1 ]; then
  python local/download_grid_corpus.py --dir $datasrc
fi

if [ $stage -le 0 ]; then 	
  local/grid_data_prep.sh $datasrc
fi

if [ $stage -le 1 ]; then
  echo "preparing dictionary and lang"
  local/chime1_prepare_dict.sh $dict || exit 1
   
  utils/prepare_lang.sh --num-sil-states 5 \
     --num-nonsil-states 3 \
     --position-dependent-phones false \
     --share-silence-phones true \
     $dict "a"  $langtmp $lang || exit 1

  local/grid_prepare_grammar.sh || exit 1
fi

if [ $stage -le 2 ]; then
  for x in test train; do
    echo "preparing features"
    rm -rf data/${x}_raw
    local/copy_data_dir.sh data/$x data/${x}_raw 
    local/make_avhubert.sh --cmd "$decode_cmd"  --nj $njfeats  --ckpt $avhubert_ckpt \
     --avhubert-path $avhubert_path --layer $encoder_layer data/${x}_raw  || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_raw   || exit 1;
    utils/fix_data_dir.sh data/${x}_raw
  done
fi

if [ $stage -le 3 ]; then
  pca_model=pca-${pca_dim}d.pt    
  pca_dir=pca
  mkdir -p $pca_dir
 
  if [[ ! -f $pca_dir/$pca_model  ||  $pca_dir/$pca_model \
	  -ot data/train_raw/feats.scp ]] ; then
    echo "Training PCA model"
    mkdir -p $pca_dir
    python local/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/train_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=$pca_train_utts \
      --interp_mode=$interp_mode --upsample_factor=$upsample_factor \
      $pca_dir/$pca_model
  fi

  for x in train test; do
    echo "preparing pca features"    
    local/make_pca_features.sh --cmd "$decode_cmd" --nj 15 \
	  --interp-mode $interp_mode --upsample-factor $upsample_factor \
	  --pca-model $pca_dir/$pca_model \
          data/${x}_raw data/${x}   || exit 1;
    steps/compute_cmvn_stats.sh data/$x   || exit 1;
    utils/fix_data_dir.sh data/$x
  done
fi

if [ $stage -le 4 ]; then
  steps/train_mono.sh --nj 8 --cmd "$train_cmd" --boost-silence 1.5 \
    data/train data/lang exp/mono
fi

if [ $stage -le 5 ]; then
  utils/mkgraph.sh  data/lang  exp/mono exp/mono/graph
  steps/decode.sh --config conf/decode.config --nj $njtest --cmd "$decode_cmd" \
    exp/mono/graph data/test exp/mono/decode  
fi

if [ $stage -le 6 ]; then
  # Get alignments from monophone system.
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" --boost-silence 1.5 \
  $numLeavesTri1  $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1

  utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph
  steps/decode.sh --config conf/decode.config --nj $njtest --cmd "$decode_cmd" \
    exp/tri1/graph data/test exp/tri1/decode
  
  # align tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri1 exp/tri1_ali
fi

if [ $stage -le 6 ]; then
  # train and decode tri2b [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $numLeavesMLLT \
    $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2b

  utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
  steps/decode.sh --config conf/decode.config --nj $njtest --cmd "$decode_cmd" \
    exp/tri2b/graph data/test exp/tri2b/decode 
fi

if [ $stage -le 7 ]; then
# Align all data with LDA+MLLT system (tri2b)
steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/train data/lang exp/tri2b exp/tri2b_ali
fi

if [ $stage -le 8 ]; then
  # Do LDA+MLLT+SAT, and decode.
  steps/train_sat.sh --cmd "$train_cmd"  $numLeavesSAT  $numGaussSAT data/train \
    data/lang exp/tri2b_ali exp/tri3b
    
  utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph
  steps/decode_fmllr.sh --config conf/decode.config --nj $njtest --cmd "$decode_cmd" \
      exp/tri3b/graph data/test exp/tri3b/decode
fi

if [ $stage -le 9 ]; then
  # Align all data with LDA+MLLT+SAT 
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      data/train data/lang exp/tri3b exp/tri3b_ali
fi

data_fmllr=data-fmllr-tri3b
gmm=exp/tri3b
if [ $stage -le 10 ]; then
  # test
  dirc=${data_fmllr}/test
  mkdir -p $dirc
  steps/nnet/make_fmllr_feats.sh --nj 4 --cmd "$train_cmd" \
     --transform-dir $gmm/decode \
     $dirc data/test $gmm $dirc/log $dirc/data
  steps/compute_cmvn_stats.sh $dirc  $dirc/log $dirc/data || exit 1;
  # train
  dirc=${data_fmllr}/train
  mkdir -p $dirc
  steps/nnet/make_fmllr_feats.sh --nj 8 --cmd "$train_cmd" \
     --transform-dir ${gmm}_ali \
     $dirc data/train $gmm $dirc/log $dirc/data
  steps/compute_cmvn_stats.sh $dirc  $dirc/log $dirc/data || exit 1;
fi

exit 0;