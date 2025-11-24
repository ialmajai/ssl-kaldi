#!/usr/bin/env bash

set -euo pipefail

stage=6
train_set=train
test_sets=dev
gmm=tri2
layer=9

num_data_reps=1  # number of reverberated copies of data to generate
                 # These will be combined with the original data.
speed_perturb=true

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

clean_data_dir=${train_set}_sp

if [ $stage -le 6 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${clean_data_dir}
  echo "$0: making mHuBERT features for low-resolution speed-perturbed data"
  local/make_mhubert.sh --cmd "run.pl" --nj 6 --layer $layer --apply-pca true --feat-dim 30 data/${clean_data_dir} || exit 1;
  steps/compute_cmvn_stats.sh data/${clean_data_dir} || exit 1;
  utils/fix_data_dir.sh data/${clean_data_dir}
fi

if [ $stage -le 7 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_si.sh --nj 20 --cmd "$train_cmd" \
    data/${clean_data_dir} data/lang $gmm_dir $ali_dir || exit 1
fi

if [ $stage -le 8 ]; then
  if [ ! -d "simulated_rirs_16k" ]; then
    # Download the simulated RIR package with 8k sampling rate
    wget --no-check-certificate http://www.openslr.org/resources/26/sim_rir_16k.zip
    unzip sim_rir_16k.zip
  fi

  python steps/data/reverberate_data_dir.py \
    --prefix "rev" \
    --rir-set-parameters "0.3, simulated_rirs_16k/smallroom/rir_list" \
    --rir-set-parameters "0.3, simulated_rirs_16k/mediumroom/rir_list" \
    --rir-set-parameters "0.3, simulated_rirs_16k/largeroom/rir_list" \
    --speech-rvb-probability 1 \
    --num-replications $num_data_reps \
    --source-sampling-rate 16000 \
    --include-original-data true \
    data/${train_set}_sp data/${clean_data_dir}_rvb${num_data_reps}
fi

if [ $stage -le 9 ]; then
  echo "$0: extract mHUBERT features w/o pca"

  for datadir in ${clean_data_dir}_rvb${num_data_reps} ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_768
    utils/data/perturb_data_dir_volume.sh data/${datadir}_768 || exit 1;
    
    local/make_mhubert.sh --nj 6 --layer $layer --compress true --apply-pca false --feat-dim 768   \
      --cmd "$train_cmd" data/${datadir}_768 || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_768 || exit 1;
    utils/fix_data_dir.sh data/${datadir}_768 || exit 1;
  done
fi

if [ $stage -le 10 ]; then
  echo "$0: extract mHuBERT features w/o pca"

  for datadir in dev ; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_768

    local/make_mhubert.sh --nj 6  --layer $layer  --feat-dim 768   \
      --cmd "$train_cmd" data/${datadir}_768 || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_768 || exit 1;
    utils/fix_data_dir.sh data/${datadir}_768 || exit 1;
  done
fi
