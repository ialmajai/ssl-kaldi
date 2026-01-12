#!/usr/bin/env bash
# Copyright  2025    Ibrahim Almajai
# Apache 2.0

set -euo pipefail

stage=0
feats_nj=8
train_set=train
test_set=dev
gmm=tri2
layer=9
pca_dim=30

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

clean_data_dir=${train_set}_sp_raw
if [ $stage -le 0 ]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set} data/${clean_data_dir}
  echo "$0: making mHuBERT features for low-resolution speed-perturbed data"
  local/make_mhubert.sh --cmd "run.pl" --nj $feats_nj --layer $layer data/${clean_data_dir} || exit 1;
  steps/compute_cmvn_stats.sh data/${clean_data_dir} || exit 1;
  utils/fix_data_dir.sh data/${clean_data_dir}
fi

if [ $stage -le 1 ]; then
  pca_model="pca-${pca_dim}d-sp.pt"    
  pca_dir="pca"
  mkdir -p $pca_dir
  if [[ ! -f $pca_dir/$pca_model  ||  $pca_dir/$pca_model \
          -ot data/${clean_data_dir}/feats.scp ]] ; then
    echo "Training PCA model"
    mkdir -p $pca_dir
    python local/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/${clean_data_dir}/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=20000 $pca_dir/$pca_model
  fi
  
  echo "preparing pca features"    
  utils/copy_data_dir.sh data/$clean_data_dir data/${train_set}_sp_pca
  rm -rf data/${train_set}_sp_pca/feats.scp data/${train_set}_sp_pca/data 
  local/make_pca_features.sh --cmd "$decode_cmd"  --nj 15  --pca-model $pca_dir/$pca_model \
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
  echo "$0: extract raw mHUBERT features"
  utils/data/perturb_data_dir_volume.sh data/${rev_data_dir} || exit 1;
  
  local/make_mhubert.sh --nj $feats_nj --layer $layer   \
    --cmd "$train_cmd" data/${rev_data_dir} || exit 1;
  steps/compute_cmvn_stats.sh data/${rev_data_dir} || exit 1;
  utils/fix_data_dir.sh data/${rev_data_dir} || exit 1;  
fi

exit 0;