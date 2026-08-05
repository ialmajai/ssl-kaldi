#!/usr/bin/env bash
# Copyright  2025    Ibrahim Almajai
# Apache 2.0

set -euo pipefail

stage=0
feats_nj=8
train_set=train
test_set=dev
gmm=tri2
ssl_model=
layer=9
pca_dim=30

num_data_reps=1  # number of reverberated copies of data to generate
                 # These will be combined with the original data.
speed_perturb=true

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

[ -z "$ssl_model" ] && echo "$0: --ssl-model is required" && exit 1

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}_raw/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

clean_data_dir=${train_set}_sp_raw
if [ $stage -le 0 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set}_raw data/${clean_data_dir}
  echo "$0: making ssl features for low-resolution speed-perturbed data"
  shared/make_ssl.sh --cmd "$train_cmd" --nj $feats_nj --ssl-model $ssl_model \
	  --layer $layer data/${clean_data_dir} || exit 1;
  steps/compute_cmvn_stats.sh data/${clean_data_dir} || exit 1;
  utils/fix_data_dir.sh data/${clean_data_dir}
fi

if [ $stage -le 1 ]; then
  # Reuse the PCA model that run.sh stage 2 fitted, rather than fitting a
  # second one on the speed-perturbed data. $gmm was trained under that model
  # and is about to align the features projected below, and eigenvector sign is
  # arbitrary, so a separately fitted basis hands it a negated low-variance
  # dimension, differently on each run.
  #
  # Refitting also measures as a no-op, at least on timit: an sp-fitted basis
  # retained 0.024 percentage points more variance of the perturbed data, with
  # the two subspaces overlapping at 0.9998. Speed perturbation rescales time
  # and barely moves the distribution of frame-wise embeddings. Not re-measured
  # on this corpus. See egs/timit/s5/NOTES.md.
  pca_model="pca-${pca_dim}d.pt"
  pca_dir="pca"
  if [ ! -f $pca_dir/$pca_model ]; then
    echo "$0: expected $pca_dir/$pca_model from run.sh stage 2" && exit 1
  fi


  echo "preparing pca features"    
  utils/copy_data_dir.sh data/$clean_data_dir data/${train_set}_sp_pca
  rm -rf data/${train_set}_sp_pca/feats.scp data/${train_set}_sp_pca/data 
  shared/make_pca_features.sh --cmd "$decode_cmd"  --nj 15  --pca-model $pca_dir/$pca_model \
        data/${clean_data_dir} data/${train_set}_sp_pca  || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_pca  || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp_pca  
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_si.sh --nj 20 --cmd "$train_cmd" \
    data/${train_set}_sp_pca data/lang $gmm_dir $ali_dir || exit 1
fi

rev_data_dir=${clean_data_dir}_rvb${num_data_reps}
if [ $stage -le 3 ]; then
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
    data/${clean_data_dir} data/${rev_data_dir}
fi

if [ $stage -le 4 ]; then
  echo "$0: extract raw ssl features"
  utils/data/perturb_data_dir_volume.sh data/${rev_data_dir} || exit 1;
  
  shared/make_ssl.sh --nj $feats_nj --ssl-model $ssl_model --layer $layer \
    --cmd "$train_cmd" data/${rev_data_dir} || exit 1;
  steps/compute_cmvn_stats.sh data/${rev_data_dir} || exit 1;
  utils/fix_data_dir.sh data/${rev_data_dir} || exit 1;  
fi

exit 0;
