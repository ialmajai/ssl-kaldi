#!/usr/bin/env bash
# Copyright 2025    Ibrahim Almajai
#
# end-to-end LF-MMI training  
# stages 0-2 are the same as "run.sh"

set -euo pipefail

stage=0
trainset=train_raw
testset=test_raw

frame_subsampling_factor=2

use_upsampling=true
upsample_factor=2
interp_mode=nearest


. ./cmd.sh 
. ./path.sh
. utils/parse_options.sh

if  $use_upsampling && [ $stage -le 3 ]; then
  for x in train test; do
    echo "interpolating raw features"
    src=data/${x}_raw
    dst=data/${x}_raw_upsampled 
    local/copy_data_dir.sh $src $dst    
    local/upsample_features.sh --cmd "$decode_cmd" --nj 15 \
      --interp-mode $interp_mode --upsample-factor $upsample_factor \
       $src $dst   || exit 1;
    steps/compute_cmvn_stats.sh $dst   || exit 1;
    utils/fix_data_dir.sh $dst
  done

fi

# required files for flat-start
echo -e '2.96\n3.00' > data/$trainset/allowed_durs.txt
if  $use_upsampling ; then
  echo -e '148\n150' > data/$trainset/allowed_lengths.txt
  trainset=${trainset}_upsampled
  testset=${testset}_upsampled
else
  echo -e '74\n75' > data/$trainset/allowed_lengths.txt
fi

if [ $stage -le 4 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh --train-set $trainset --test-set $testset \
	  --frame-subsampling-factor $frame_subsampling_factor
fi
