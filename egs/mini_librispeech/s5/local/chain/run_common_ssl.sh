#!/usr/bin/env bash
set -euo pipefail

stage=12
train_set=train_clean_5
test_sets="dev_clean_2"
gmm=tri3b

pca_dim=30
nnet3_affix=
ssl_model=
feats_nj=
encoder_layer= 
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}_raw/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if [ $stage -le 12 ]; then
  compute_mode=$(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=compute_mode --format=csv,noheader | head -n1 || true)
  if [ "$compute_mode" == "Exclusive_Process" ]; then
    echo "Feature extraction requires GPU compute mode to be set to default"
    echo "run: sudo nvidia-smi -c 0"
    exit 1
  fi
  # Although the nnet will be trained by high resolution data, we still have to
  # perturb the normal data to get the alignment _sp stands for speed-perturbed
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/${train_set}_raw data/${train_set}_sp_raw

  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_raw || exit 1;

  echo "$0: making SSL features for low-resolution speed-perturbed data"
  shared/make_ssl.sh --cmd "run.pl" --nj $feats_nj --layer $encoder_layer \
	  --ssl-model $ssl_model data/${train_set}_sp_raw || exit 1;
  steps/compute_cmvn_stats.sh data/${train_set}_sp_raw || exit 1;
  utils/fix_data_dir.sh data/${train_set}_sp_raw
fi

if [ $stage -le 13 ]; then
  # Reuse the PCA model that run.sh stage 4 fitted on the unperturbed data,
  # rather than fitting a second one on the speed-perturbed set. Refitting
  # measures as a no-op (0.024 percentage points more variance retained on the
  # perturbed data, subspaces overlapping at 0.9998) and a separately fitted
  # basis hands the GMM a sign-flipped low-variance dimension, differently on
  # each run, because eigenvector sign is arbitrary. The GMM that aligns these
  # features was trained under this model. See egs/timit/s5/NOTES.md.
  pca_model="pca-${pca_dim}d.pt"
  pca_dir="pca"
  if [ ! -f $pca_dir/$pca_model ]; then
    echo "$0: expected $pca_dir/$pca_model from run.sh stage 4" && exit 1
  fi
  for part in train_clean_5_sp; do
    echo "preparing pca features"    
    utils/copy_data_dir.sh data/${part}_raw data/${part}_pca
    shared/make_pca_features.sh --cmd "$decode_cmd"  --nj 15  --pca-model $pca_dir/$pca_model \
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
