#!/usr/bin/env bash

data=./corpus/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

stage=0

# choose either base or large model and layer to extract features
# base model
# model_type="facebook/hubert-base-ls960"
# encoder_layer=9
# feats_nj=8

# large model
model_type="facebook/hubert-large-ll60k"
encoder_layer=12
feats_nj=4

pca_dim=30

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail


echo "Using model: $model_type and layer: $encoder_layer for feature extraction"

mkdir -p data

if [ $stage -le 0 ]; then
  mkdir -p $data
  for part in dev-clean-2 train-clean-5; do
    local/download_and_untar.sh $data $data_url $part
  done
fi

if [ $stage -le 1 ]; then
  local/download_lm.sh $lm_url $data data/local/lm
fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean-2 train-clean-5; do
    # use underscore-separated names in data directories.
    local/data_prep.sh corpus/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  local/prepare_dict.sh  --nj 30 --stage 3 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
fi

if [ $stage -le 3 ]; then
  compute_mode=`nvidia-smi --query-gpu=compute_mode --format=csv,noheader`
  if [ "$compute_mode" == "Exclusive_Process" ]; then

    echo "Feature extraction requires GPU compute mode to be set to default"
    echo "run: sudo nvidia-smi -c 0"
    exit 1
  fi

  for part in dev_clean_2 train_clean_5; do
    echo "preparing features"
    utils/copy_data_dir.sh data/$part data/${part}_raw 
    local/make_hubert.sh --cmd "$train_cmd" --nj $feats_nj --model-type $model_type \
	    --layer $encoder_layer data/${part}_raw 
    steps/compute_cmvn_stats.sh data/${part}_raw 
  done  
fi

if [ $stage -le 4 ]; then
  pca_model="pca-${pca_dim}d.pt"    
  pca_dir="pca"
  mkdir -p $pca_dir
  if [[ ! -f $pca_dir/$pca_model  ||  $pca_dir/$pca_model \
          -ot data/train_clean_5_raw/feats.scp ]] ; then
    echo "Training PCA model"
    mkdir -p $pca_dir
    python local/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/train_clean_5_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=1500 $pca_dir/$pca_model
  fi
  for part in dev_clean_2 train_clean_5; do
    echo "preparing pca features"    
    utils/copy_data_dir.sh data/$part data/${part}_pca
    local/make_pca_features.sh --cmd "$decode_cmd"  --nj 15 \
        --pca-model $pca_dir/$pca_model \
          data/${part}_raw data/${part}_pca   || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_pca  || exit 1;
    utils/fix_data_dir.sh data/${part}_pca
  done 
fi

# train a monophone system
if [ $stage -le 5 ]; then
  utils/subset_data_dir.sh --shortest data/train_clean_5_pca 500 data/train_500short

  steps/train_mono.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_500short data/lang_nosp exp/mono

  steps/align_si.sh --boost-silence 1.25 --nj 5 --cmd "$train_cmd" \
    data/train_clean_5_pca data/lang_nosp exp/mono exp/mono_ali_train_clean_5
fi

if [ $stage -le 6 ]; then
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                   exp/mono exp/mono/graph_tgsmall
    for testset in dev_clean_2_pca; do
      steps/decode.sh --nj 20 --cmd "$decode_cmd" exp/mono/graph_tgsmall \
        data/$testset exp/mono/decode_tgsmall_$testset
      steps/lmrescore.sh --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tgmed} \
                       data/$testset exp/mono/decode_{tgsmall,tgmed}_$testset
      steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_nosp_test_{tgsmall,tglarge} \
      data/$testset exp/mono/decode_{tgsmall,tglarge}_$testset
    done
fi
#delta + delta-delta triphone
if [ $stage -le 7 ]; then
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 data/train_clean_5_pca data/lang_nosp exp/mono_ali_train_clean_5 exp/tri1

  steps/align_si.sh --nj 5 --cmd "$train_cmd" \
    data/train_clean_5_pca data/lang_nosp exp/tri1 exp/tri1_ali_train_clean_5
fi

# Train LDA+MLLT system
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
    data/train_clean_5_pca data/lang_nosp exp/tri1_ali_train_clean_5 exp/tri2b

  steps/align_si.sh  --nj 5 --cmd "$train_cmd" --use-graphs true \
    data/train_clean_5_pca data/lang_nosp exp/tri2b exp/tri2b_ali_train_clean_5
fi

# Train LDA+MLLT+SAT
if [ $stage -le 9 ]; then
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
    data/train_clean_5_pca data/lang_nosp exp/tri2b_ali_train_clean_5 exp/tri3b
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 10 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/train_clean_5_pca data/lang_nosp exp/tri3b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
    "<UNK>" data/local/lang_tmp data/lang

  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
fi

if [ $stage -le 11 ]; then
  # decode using the tri3b model
  utils/mkgraph.sh data/lang_test_tgsmall \
                   exp/tri3b exp/tri3b/graph_tgsmall
  for test in  dev_clean_2_pca ; do
    steps/decode_fmllr.sh --nj 10  --cmd "$decode_cmd" \
                          exp/tri3b/graph_tgsmall data/$test \
                          exp/tri3b/decode_tgsmall_$test
    steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
                       data/$test exp/tri3b/decode_{tgsmall,tgmed}_$test
    steps/lmrescore_const_arpa.sh \
      --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      data/$test exp/tri3b/decode_{tgsmall,tglarge}_$test
  done
fi

if [ $stage -le 12 ]; then
  echo "$0: TDNN training started"
  local/chain/run_common_hubert.sh --stage 13 \
    --feats-nj $feats_nj \
    --model-type $model_type \
    --encoder-layer $encoder_layer \
    --pca-dim $pca_dim

  local/chain/run_tdnn_hubert.sh --stage 15 \
    --gmm tri3b \
    --affix "1a"
fi


