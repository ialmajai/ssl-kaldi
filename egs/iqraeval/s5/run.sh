#!/usr/bin/env bash
#
# IqraEval Phone Recognition with mHuBERT SSL Features
#
# Expected directory structure:
#   $IqraEvalData/
#     CV-Ar/{train,dev}/{wav,transcripts}/
#     TTS/{train,dev}/{wav,transcripts}/
#



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

feats_nj=6
train_nj=16
decode_nj=16


#Modify dataset path
IqraEvalData=/data/git/interspeech_IqraEval/sws_data

set -euo pipefail  # Exit on error, undefined variables

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
  if [ ! -f pca_mhubert/pca-l$encoder_layer-${pca_dim}d.pt ]; then
    #Calculate PCA 
    echo  "PCA calculation: encoder layer no. $encoder_layer"
    python local/pca_mhubert.py  --layer $encoder_layer -d $pca_dim
  fi
fi


if [ $stage -le 2 ]; then
  # Feature Extration & CMVN
  ssl_dir=mhubert

  for x in dev train ; do
    local/make_mhubert.sh  --cmd "$train_cmd" --apply-pca true --feat-dim $pca_dim \
       --nj $feats_nj --layer $encoder_layer data/${x} exp/make_mhubert/$x $ssl_dir
    steps/compute_cmvn_stats.sh data/$x exp/make_mhubert/$x $ssl_dir
  done

fi

if [ $stage -le 3 ]; then

  steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

  utils/mkgraph.sh data/lang_test_trigram exp/mono exp/mono/graph

  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/mono/graph data/dev exp/mono/decode_dev_tg

fi

# Deltas + Delta-Deltas 
if [ $stage -le 4 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali

  # Train tri1, which is deltas + delta-deltas, on train data.
  steps/train_deltas.sh --cmd "$train_cmd" \
     $numLeavesTri1 $numGaussTri1 data/train data/lang exp/mono_ali exp/tri1

  utils/mkgraph.sh data/lang_test_trigram exp/tri1 exp/tri1/graph

  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
     exp/tri1/graph data/dev exp/tri1/decode_dev_tg
fi

# tri2 : LDA + MLLT 
if [ $stage -le 5 ]; then
  steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
     data/train data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT data/train data/lang exp/tri1_ali exp/tri2

    utils/mkgraph.sh data/lang_test_trigram exp/tri2 exp/tri2/graph

  steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
    exp/tri2/graph data/dev exp/tri2/decode_dev_tg
fi

echo "========================================="
echo "Script completed: $(date)"
echo "========================================="

