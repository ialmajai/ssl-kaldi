#!/usr/bin/env bash
set -euo pipefail

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

if [ $stage -le 12 ]; then
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set}_raw data/${train_set}_sp_raw

  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_raw || exit 1;

  echo "$0: making HuBERT features for low-resolution speed-perturbed data"
  local/make_hubert.sh --cmd "run.pl" --nj 4 --layer $encoder_layer \
	   data/${train_set}_sp_raw || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_raw || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp_raw
fi

if [ $stage -le 13 ]; then
  pca_model="pca-${pca_dim}d-sp.pt"    
  pca_dir="pca"
  mkdir -p $pca_dir
  if [ ! -f $pca_dir/$pca_model ]; then
    echo "Training PCA model"
    mkdir -p $pca_dir
    python local/pca.py  --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/${train_set}_sp_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=1500 $pca_dir/$pca_model
  fi
  for part in train_clean_5_sp; do
    echo "preparing pca features"    
    utils/copy_data_dir.sh data/${part}_raw data/${part}_pca
    local/make_pca_features.sh --cmd "$decode_cmd"  --nj 15  --pca-model $pca_dir/$pca_model \
          data/${part}_raw data/${part}_pca  || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_pca  || exit 1;
    utils/fix_data_dir.sh data/${part}_pca
  done 
fi

if [ $stage -le 14 ]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/${train_set}_sp_pca data/lang $gmm_dir $ali_dir || exit 1
fi

exit 0


