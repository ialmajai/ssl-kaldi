#!/usr/bin/env bash
# Copyright  2025    Ibrahim Almajai
# Apache 2.0
# IqraEval Phone Recognition with mHuBERT SSL Features

set -euo pipefail 

stage=0
encoder_layer=9
pca_dim=30

. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

# GMM parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000

feats_nj=8
train_nj=16
decode_nj=16

#Modify dataset path
IqraEvalData=/data/git/interspeech_IqraEval/sws_data

echo "========================================="
echo "IqraEval Phone Recognition - GMM Pipeline"
echo "Started: $(date)"
echo "========================================="

if [ $stage -le 0 ]; then 

  local/iqra_data_prep.sh  $IqraEvalData || exit 1
  # Verify data directories were created
  for x in train dev; do
    utils/fix_data_dir.sh data/$x || exit 1	  
    utils/validate_data_dir.sh --no-feats data/$x  || exit 1
  done
  local/iqra_prepare_dict.sh || exit 1
  utils/validate_dict_dir.pl  data/local/dict || exit 1
  utils/prepare_lang.sh --sil-prob 0.1 --position-dependent-phones false \
    data/local/dict "sil" data/local/lang_tmp data/lang || exit 1
  local/iqra_prepare_lms.sh || exit 1
fi

if [ $stage -le 1 ]; then
  # Feature Extration & CMVN
  for x in dev train ; do
    utils/copy_data_dir.sh data/$x data/${x}_raw
    local/make_mhubert.sh  --cmd "$train_cmd" \
       --nj $feats_nj --layer $encoder_layer data/${x}_raw 
    steps/compute_cmvn_stats.sh data/${x}_raw 
  done
fi

exit 0

if [ $stage -le 2 ]; then
  pca_model="pca-${pca_dim}d.pt"    
  pca_dir="pca"
  mkdir -p $pca_dir
  if [[ ! -f $pca_dir/$pca_model  ||  $pca_dir/$pca_model \
          -ot data/train_raw/feats.scp ]] ; then
    echo "Training PCA model"
    mkdir -p $pca_dir
    python local/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/train_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=20000 $pca_dir/$pca_model
  fi
  for x in dev train; do
    echo "preparing pca features"    
    utils/copy_data_dir.sh data/$x data/${x}_pca
    local/make_pca_features.sh --cmd "$decode_cmd"  --nj 15  --pca-model $pca_dir/$pca_model \
          data/${x}_raw data/${x}_pca   || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_pca   || exit 1;
    utils/fix_data_dir.sh data/${x}_pca
  done 
fi

# mono: PCA features
if [ $stage -le 3 ]; then
  utils/subset_data_dir.sh data/train_pca 5000 data/train_pca_5k
  steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train_pca_5k data/lang exp/mono

  utils/mkgraph.sh data/lang_test_trigram exp/mono exp/mono/graph
  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/dev_pca exp/mono/decode_dev_tg
fi

# tri1: Δ + ΔΔ
if [ $stage -le 4 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    data/train_pca data/lang exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train_pca data/lang exp/mono_ali exp/tri1

  utils/mkgraph.sh data/lang_test_trigram exp/tri1 exp/tri1/graph
  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/dev_pca exp/tri1/decode_dev_tg
fi

# tri2: LDA + MLLT 
if [ $stage -le 5 ]; then
  steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
     data/train_pca data/lang exp/tri1 exp/tri1_ali
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT data/train_pca data/lang exp/tri1_ali exp/tri2

  utils/mkgraph.sh data/lang_test_trigram exp/tri2 exp/tri2/graph
  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
    exp/tri2/graph data/dev_pca exp/tri2/decode_dev_tg
fi

# DNN training and decoding
if [ $stage -le 6 ]; then
  local/chain/run_common.sh --stage 0 --gmm tri2 --layer $encoder_layer \
                            --pca-dim $pca_dim --num-data-reps 1
  local/chain/run_tdnn_mhubert_mono_rvb.sh --stage 0 --gmm tri2 --num-data-reps 1
fi

if [ $stage -le 7 ]; then
  echo "========================================="
  echo "IqraEval Phone Recognition - GMM Pipeline"
  echo "Completed: $(date)"
  echo "========================================="
fi





