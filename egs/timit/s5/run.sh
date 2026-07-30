#!/usr/bin/env bash
# Copyright  2026    Ibrahim Almajai
# Apache 2.0
# TIMIT phone recognition with SSL features instead of MFCCs.
#
# Same pipeline as Kaldi's egs/timit/s5 (see run-mfcc.sh), except that the
# acoustic features are frame-level embeddings from a frozen pretrained SSL
# model:
#   data/<set>_raw  768-dim (base) / 1024-dim (large) SSL features, for the
#                   chain TDNN-F
#   data/<set>_pca  PCA-reduced SSL features, for the GMM stages
# Training uses 48 phones, scoring maps them to the standard 39 classes.

set -euo pipefail

stage=0
stop_stage=100   # last stage to run; e.g. --stop-stage 5 stops after tri2

# SSL feature extraction. mms-300m layer 14 is the best configuration measured
# here (10.19% PER); hubert-base-ls960 layer 9 is 11.43% at a third of the
# extraction cost, and hubert-large-ll60k layer 14 is no better than base.
# ssl_model="facebook/hubert-base-ls960"
# encoder_layer=9
ssl_model="facebook/mms-300m"
encoder_layer=14
pca_dim=30

# Path to the TIMIT corpus, i.e. the directory containing TRAIN/ and TEST/
# (LDC93S1). Override on the command line: ./run.sh --timit /path/to/TIMIT
timit=/export/corpora5/LDC/LDC93S1/timit/TIMIT

# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000

feats_nj=4
train_nj=30
decode_nj=8

# Chain training job schedule; set both to 1 for a single GPU.
chain_jobs_initial=2
chain_jobs_final=4

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

test_sets="dev test"

echo "Using model: $ssl_model and layer: $encoder_layer for feature extraction"

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "============================================================================"
  echo "                Data & Lexicon & Language Preparation                       "
  echo "============================================================================"

  local/timit_data_prep.sh $timit || exit 1

  local/timit_prepare_dict.sh || exit 1

  # Caution below: we remove optional silence by setting "--sil-prob 0.0",
  # in TIMIT the silence appears also as a word in the dictionary and is scored.
  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false \
    --num-sil-states 3 data/local/dict "sil" data/local/lang_tmp data/lang || exit 1

  local/timit_format_data.sh || exit 1
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "============================================================================"
  echo "                SSL Feature Extraction & CMVN                                "
  echo "============================================================================"

  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if [ "$compute_mode" == "Exclusive_Process" ]; then
    echo "Feature extraction requires GPU compute mode to be set to default"
    echo "run: sudo nvidia-smi -c 0"
    exit 1
  fi

  for x in train $test_sets; do
    utils/copy_data_dir.sh data/$x data/${x}_raw
    shared/make_ssl.sh --cmd "$train_cmd" --nj $feats_nj --ssl-model $ssl_model \
      --layer $encoder_layer data/${x}_raw
    steps/compute_cmvn_stats.sh data/${x}_raw
  done
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "============================================================================"
  echo "                PCA Features for the GMM Stages                             "
  echo "============================================================================"

  pca_model="pca-${pca_dim}d.pt"
  pca_dir="pca"
  mkdir -p $pca_dir
  if [[ ! -f $pca_dir/$pca_model || $pca_dir/$pca_model \
          -ot data/train_raw/feats.scp ]] ; then
    echo "Training PCA model"
    python shared/pca.py --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/train_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=1500 $pca_dir/$pca_model
  fi
  for x in train $test_sets; do
    echo "preparing pca features"
    utils/copy_data_dir.sh data/$x data/${x}_pca
    shared/make_pca_features.sh --cmd "$decode_cmd" --nj $feats_nj \
      --pca-dim $pca_dim --pca-model $pca_dir/$pca_model \
      data/${x}_raw data/${x}_pca || exit 1;
    steps/compute_cmvn_stats.sh data/${x}_pca || exit 1;
    utils/fix_data_dir.sh data/${x}_pca
  done
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "============================================================================"
  echo "                     MonoPhone Training & Decoding                          "
  echo "============================================================================"

  steps/train_mono.sh --nj "$train_nj" --cmd "$train_cmd" \
    data/train_pca data/lang exp/mono

  utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph
  for x in $test_sets; do
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
      exp/mono/graph data/${x}_pca exp/mono/decode_$x
  done
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "============================================================================"
  echo "           tri1 : Deltas + Delta-Deltas Training & Decoding                 "
  echo "============================================================================"

  steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
    data/train_pca data/lang exp/mono exp/mono_ali

  steps/train_deltas.sh --cmd "$train_cmd" \
    $numLeavesTri1 $numGaussTri1 data/train_pca data/lang exp/mono_ali exp/tri1

  utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph
  for x in $test_sets; do
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
      exp/tri1/graph data/${x}_pca exp/tri1/decode_$x
  done
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  echo "============================================================================"
  echo "                 tri2 : LDA + MLLT Training & Decoding                      "
  echo "============================================================================"

  steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
    data/train_pca data/lang exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    $numLeavesMLLT $numGaussMLLT data/train_pca data/lang exp/tri1_ali exp/tri2

  utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph
  for x in $test_sets; do
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" \
      exp/tri2/graph data/${x}_pca exp/tri2/decode_$x
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "============================================================================"
  echo "              tri3 : LDA + MLLT + SAT Training & Decoding                   "
  echo "============================================================================"

  steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" --use-graphs true \
    data/train_pca data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/train_pca data/lang exp/tri2_ali exp/tri3

  utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph
  for x in $test_sets; do
    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" \
      exp/tri3/graph data/${x}_pca exp/tri3/decode_$x
  done
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  echo "============================================================================"
  echo "         Speed-perturbed features & alignments for chain training           "
  echo "============================================================================"

  local/chain/run_common_ssl.sh --stage 0 \
    --ssl-model $ssl_model \
    --encoder-layer $encoder_layer \
    --feats-nj $feats_nj \
    --pca-dim $pca_dim \
    --gmm tri3
fi

if [ $stage -le 8 ] && [ $stop_stage -ge 8 ]; then
  echo "============================================================================"
  echo "                    Chain TDNN-F Training & Decoding                        "
  echo "============================================================================"

  local/chain/run_tdnn_ssl.sh --stage 0 \
    --gmm tri3 \
    --decode-nj $decode_nj \
    --num-jobs-initial $chain_jobs_initial \
    --num-jobs-final $chain_jobs_final \
    --affix "1a"
fi

echo "============================================================================"
echo "                    Getting Results [see RESULTS file]                      "
echo "============================================================================"
bash RESULTS dev
bash RESULTS test

echo "Finished successfully on" `date`
exit 0
