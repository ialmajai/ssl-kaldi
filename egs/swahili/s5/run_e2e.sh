#!/usr/bin/env bash
# Copyright 2025    Ibrahim Almajai
# end-to-end LF-MMI training  


set -euo pipefail
stage=0
trainset=train
frame_subsampling_factor=2

. ./cmd.sh 
. ./path.sh
. utils/parse_options.sh

#ssl_model="ajesujoba/AfriHuBERT"
ssl_model="utter-project/mHuBERT-147"
encoder_layer=9
feats_nj=8

echo "Using model: $ssl_model and layer: $encoder_layer for feature extraction"

#download all repo to build an ASR of Swahili

#!/usr/bin/env bash

# initialization PATH
. ./path.sh  || die "path.sh expected";
# initialization commands
. ./cmd.sh

#download Swahili dataset
if [ ! -d "asr_swahili" ]; then
  # https://www.openslr.org/25/
  wget https://openslr.trmal.net/resources/25/data_broadcastnews_sw.tar.bz2
  tar -xvf data_broadcastnews_sw.tar.bz2
  mv data_broadcastnews_sw asr_swahili
  rm data_broadcastnews_sw.tar.bz2
fi

if [ $stage -le 0 ]; then
  # Data preparation
  local/prepare_data.sh train test

  local/prepare_dict.sh
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
  local/prepare_lm.sh
fi

if [ $stage -le 1 ]; then
  # Testset feature extraction
  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if [ "$compute_mode" == "Exclusive_Process" ]; then

    echo "Feature extraction requires GPU compute mode to be set to default"
    echo "run: sudo nvidia-smi -c 0"
    exit 1
  fi

  shared/make_ssl.sh  --cmd "$train_cmd" --ssl-model "$ssl_model" \
       --nj $feats_nj --layer $encoder_layer data/test
  steps/compute_cmvn_stats.sh data/test
fi

if [ $stage -le 2 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command
  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  sed -i  "s:\t: :"  data/train/text
  utils/data/perturb_speed_to_allowed_lengths.py --frame-shift 20 \
              --frame-subsampling-factor $frame_subsampling_factor \
              12 data/${trainset} \
              data/${trainset}_spe2e
  # creating utt2uniq does not work for such a small dataset
  # cat data/${trainset}_spe2e/utt2dur | \
  #   awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spe2e
fi

if [ $stage -le 3 ]; then
  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if [ "$compute_mode" == "Exclusive_Process" ]; then
    echo "Feature extraction requires GPU compute mode to be set to default"
    echo "run: sudo nvidia-smi -c 0"
    exit 1
  fi
    shared/make_ssl.sh --cmd "$train_cmd" --nj $feats_nj --ssl-model "$ssl_model" \
	    --layer $encoder_layer data/${trainset}_spe2e
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e   
fi

if [ $stage -le 4 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh --affix 1a \
    --frame-subsampling-factor $frame_subsampling_factor
fi
