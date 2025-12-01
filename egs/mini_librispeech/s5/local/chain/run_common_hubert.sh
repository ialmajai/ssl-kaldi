#!/usr/bin/env bash

set -euo pipefail

# This script is called from local/nnet3/run_tdnn.sh and
# local/chain/run_tdnn.sh (and may eventually be called by more
# scripts).  It contains the common feature preparation and


stage=0
train_set=train_clean_5
test_sets="dev_clean_2"
gmm=tri3b
encoder_layer=9
pca_dim=30


nnet3_affix=

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 10 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${train_set}_sp
  echo "$0: making HuBERT features for low-resolution speed-perturbed data"
  local/make_hubert.sh --cmd "run.pl" --nj 4 --layer $encoder_layer --feat-dim ${pca_dim} \
	  --apply-pca true data/${train_set}_sp || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [ $stage -le 11 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 12 ]; then  
  echo "$0: creating high-resolution HUBERT features"
  hubertdir=data/${train_set}_sp_raw/data

  for datadir in ${train_set}_sp ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_raw
  done

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_raw || exit 1;

  for datadir in ${train_set}_sp ${test_sets}; do
    local/make_hubert.sh --nj 4  --layer $encoder_layer \
      --cmd "$train_cmd" data/${datadir}_raw || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_raw || exit 1;
    utils/fix_data_dir.sh data/${datadir}_raw || exit 1;
  done
fi

exit 0


