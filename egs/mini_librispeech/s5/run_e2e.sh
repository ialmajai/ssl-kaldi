#!/usr/bin/env bash
# Copyright 2017    Hossein Hadian
#           2025    Ibrahim Almajai

# end-to-end LF-MMI training  
# stages 0-2 are the same as "run.sh"

set -euo pipefail

stage=0
encoder_layer=9
trainset=train_clean_5_raw
frame_subsampling_factor=2
. ./cmd.sh 
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 3 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command

  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  utils/data/perturb_speed_to_allowed_lengths.py --frame-length 20 \
                                              --frame-shift 20 \
                                              --frame-subsampling-factor ${frame_subsampling_factor} \
                                                12 data/${trainset} \
                                                data/${trainset}_spe2e
  cat data/${trainset}_spe2e/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spe2e
fi

if [ $stage -le 4 ]; then

    featdir=feats-spe2e
    local/make_hubert.sh --cmd "$train_cmd" --nj 4 \
	    --layer $encoder_layer data/${trainset}_spe2e exp/make_hubert/${trainset}_spe2e $featdir
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e exp/make_hubert/${trainset}_spe2e $featdir   
fi

if [ $stage -le 5 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh
fi
