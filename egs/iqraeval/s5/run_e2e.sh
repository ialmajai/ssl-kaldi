#!/usr/bin/env bash
# Copyright 2017    Hossein Hadian
#           2025    Ibrahim Almajai

# end-to-end LF-MMI training  
# stages 0-1 are the same as "run.sh"

set -euo pipefail

stage=0
encoder_layer=9
trainset=train
frame_subsampling_factor=1
spacing_factor=30

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

if [ $stage -le 2 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command

  utils/data/perturb_speed_to_allowed_lengths.py --frame-length 20 \
                                --frame-shift 20 \
                                --frame-subsampling-factor ${frame_subsampling_factor} \
                                $spacing_factor data/${trainset} \
                                data/${trainset}_spe2e_raw
  cat data/${trainset}_spe2e_raw/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_raw/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spe2e_raw
fi

if [ $stage -le 3 ]; then
    featdir=feats-spe2e_raw 
    local/make_mhubert.sh --cmd "$train_cmd" --nj 4 \
	    --layer $encoder_layer data/${trainset}_spe2e_raw exp/make_mhubert/${trainset}_spe2e_raw $featdir
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e_raw exp/make_mhubert/${trainset}_spe2e_raw $featdir   
fi

if [ $stage -le 4 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh
fi
