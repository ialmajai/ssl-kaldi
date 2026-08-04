#!/usr/bin/env bash
# Copyright  2026    Ibrahim Almajai
# Apache 2.0

# Prepares the speed-perturbed training data for chain training: raw SSL
# features (nnet input), PCA features (GMM alignments), and the tri3
# alignments themselves.

set -euo pipefail

stage=0
train_set=train
gmm=tri3
nnet3_affix=

ssl_model=
encoder_layer=
feats_nj=4
pca_dim=30
align_nj=30

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

[ -z "$ssl_model" ] && echo "$0: --ssl-model is required" && exit 1
[ -z "$encoder_layer" ] && echo "$0: --encoder-layer is required" && exit 1

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}_raw/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 0 ]; then
  echo "$0: preparing directory for speed-perturbed data"
  # This stage owns data/${train_set}_sp_raw and regenerates it from scratch.
  # Removing it first lets the recipe be re-run with a different SSL model:
  # perturb_data_dir_speed_3way.sh refuses to run if feats.scp is already there.
  rm -rf data/${train_set}_sp_raw
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set}_raw data/${train_set}_sp_raw
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_raw || exit 1;

  echo "$0: making SSL features for the speed-perturbed data"
  shared/make_ssl.sh --cmd "$train_cmd" --nj $feats_nj --layer $encoder_layer \
    --ssl-model $ssl_model data/${train_set}_sp_raw || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_raw || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp_raw
fi

if [ $stage -le 1 ]; then
  pca_model="pca-${pca_dim}d-sp.pt"
  pca_dir="pca"
  mkdir -p $pca_dir
  if [[ ! -f $pca_dir/$pca_model || $pca_dir/$pca_model \
          -ot data/${train_set}_sp_raw/feats.scp ]] ; then
    echo "Training PCA model"
    python shared/pca.py --pca_dim=$pca_dim --mode=train \
      --feats_scp=data/${train_set}_sp_raw/feats.scp \
      --pca_model=$pca_dir/$pca_model \
      --max_utts=1500 $pca_dir/$pca_model
  fi

  echo "preparing pca features"
  utils/copy_data_dir.sh data/${train_set}_sp_raw data/${train_set}_sp_pca
  rm -rf data/${train_set}_sp_pca/feats.scp data/${train_set}_sp_pca/data
  shared/make_pca_features.sh --cmd "$decode_cmd" --nj $feats_nj \
    --pca-dim $pca_dim --pca-model $pca_dir/$pca_model \
    data/${train_set}_sp_raw data/${train_set}_sp_pca || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_pca || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp_pca
fi

if [ $stage -le 2 ]; then
  echo "$0: aligning the speed-perturbed data with $gmm_dir"
  steps/align_fmllr.sh --nj $align_nj --cmd "$train_cmd" \
    data/${train_set}_sp_pca data/lang $gmm_dir $ali_dir || exit 1
fi

exit 0
