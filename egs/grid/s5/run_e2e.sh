#!/usr/bin/env bash
# Copyright 2025    Ibrahim Almajai
#

# end-to-end LF-MMI training  
# stages 0-2 are the same as "run.sh"

set -euo pipefail

stage=0

trainset=train_raw
frame_subsampling_factor=1
. ./cmd.sh 
. ./path.sh
. utils/parse_options.sh

# required files for flat-start
echo '2.96
3.00' > data/$trainset/allowed_durs.txt
echo '74
75' > data/$trainset/allowed_lengths.txt

if [ $stage -le 3 ]; then
  echo "$0: calling the flat-start chain recipe..."
  local/chain/e2e/run_tdnn_flatstart.sh --train-set $trainset \
	  --frame-subsampling-factor $frame_subsampling_factor
fi
