#!/usr/bin/env bash

stage=1


. ./cmd.sh || exit 1;
. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000

feats_nj=16
train_nj=16
decode_nj=16
encoder_layer=12

#Modify dataset path
IqraEvalData=/data/git/interspeech_IqraEval/s3prl/s3prl/sws_data


if [ $stage -le 0 ]; then 

  local/iqra_data_prep.sh  $IqraEvalData

  local/iqra_prepare_dict.sh 

  utils/prepare_lang.sh --sil-prob 0.1 --position-dependent-phones false \
    data/local/dict "sil" data/local/lang_tmp data/lang

  local/iqra_prepare_lms.sh

fi


if [ $stage -le 1 ]; then

  # Now make MFCC features.
  mfccdir=mfcc

  for x in train dev ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $feats_nj data/$x exp/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
  done

fi


if [ $stage -le 2 ]; then

 # steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train data/lang exp/mono

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

# LDA + MLLT 
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

exit 0

