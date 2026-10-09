#!/usr/bin/env bash
# Copyright 2025, author: Ibrahim Almajai
# Apache 2.0
#
# Extract several encoder layers in ONE forward pass, into one data dir per
# layer. For a sweep this replaces N passes with a single pass to the deepest
# layer: probing 12,16,18 costs 18 layer-evaluations instead of 46.
#
# Usage:
#   shared/make_ssl_layers.sh [options] <src-data-dir> <dest-pattern>
# e.g.
#   shared/make_ssl_layers.sh --layers 12,16,18 --ssl-model facebook/mms-300m \
#       data/train 'data/train_keep_l{layer}_raw'
#
# The dest pattern must contain the literal {layer}. Note the single quotes:
# {layer} is substituted by python, not by the shell.
#
# Features are written compressed directly by kaldiio (compression_method=2 is
# Kaldi's kSpeechFeature, the same codec copy-feats --compress=true uses), so
# unlike make_ssl.sh there is no copy-feats stage to pipe through.

nj=4
cmd=run.pl
layers=
ssl_model="facebook/hubert-base-ls960"
final_layer_norm=keep
max_failures=0
allow_missing=false

echo "$0 $*"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 2 ] || [ -z "$layers" ]; then
  cat >&2 <<EOF
Usage: $0 --layers 12,16,18 [options] <src-data-dir> <dest-pattern>
 e.g.: $0 --layers 12,16,18 data/train 'data/train_l{layer}_raw'
 The dest pattern must contain {layer}; quote it so the shell leaves it alone.
EOF
  exit 1
fi

src=$1
destpat=$2

case "$destpat" in
  *'{layer}'*) ;;
  *) echo "$0: dest pattern must contain {layer}: $destpat"; exit 1 ;;
esac

name=$(basename $src)
scp=$src/wav.scp
[ -f $scp ] || { echo "$0: no such file $scp"; exit 1; }
utils/validate_data_dir.sh --no-text --no-feats $src || exit 1

layer_list=$(echo $layers | tr ',' ' ')

# One data dir per layer, each a copy of the source.
for L in $layer_list; do
  d=${destpat//\{layer\}/$L}
  rm -rf $d
  utils/copy_data_dir.sh $src $d || exit 1
  rm -f $d/feats.scp
  mkdir -p $d/data $d/log || exit 1
done

# Logs and split scps live under the first layer's dir; the archives go to each.
logdir=${destpat//\{layer\}/$(echo $layer_list | awk '{print $1}')}/log

split_scps=
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/wav_${name}.$n.scp"
done
utils/split_scp.pl $scp $split_scps || exit 1

# {layer} reaches python untouched; JOB reaches run.pl untouched.
out_template="ark,scp:$PWD/${destpat}/data/raw_ssl_${name}.JOB.ark,$PWD/${destpat}/data/raw_ssl_${name}.JOB.scp"

$cmd JOB=1:$nj $logdir/make_ssl_layers_${name}.JOB.log \
  python shared/compute_ssl_feats.py \
    --layers $layers --ssl-model $ssl_model \
    --final-layer-norm $final_layer_norm --max-failures $max_failures \
    --out-template "$out_template" \
    --write-utt2dur ark,t:$logdir/utt2dur.JOB \
    --write-utt2num-frames $logdir/utt2num_frames.JOB \
    scp,p:$logdir/wav_${name}.JOB.scp ignored || exit 1

if grep -qi "^aborting after" $logdir/make_ssl_layers_${name}.*.log 2>/dev/null; then
  echo "$0: ERROR: feature extraction aborted."
  grep -hiE "Failed to process|^aborting after" $logdir/make_ssl_layers_${name}.*.log \
    | sed "s|^|$0:   |"
  echo "$0: If it is CUDA OOM, re-run with a smaller --nj (currently $nj), or"
  echo "$0: with fewer --layers: every layer is held in memory per utterance."
  exit 1
fi

for L in $layer_list; do
  d=${destpat//\{layer\}/$L}
  for n in $(seq $nj); do
    cat $d/data/raw_ssl_${name}.$n.scp || exit 1
  done > $d/feats.scp || exit 1
  cat $logdir/utt2num_frames.* > $d/utt2num_frames || exit 1
  cat $logdir/utt2dur.*        > $d/utt2dur        || exit 1
  echo 0.02 > $d/frame_shift

  nf=$(wc -l < $d/feats.scp)
  nu=$(wc -l < $d/utt2spk)
  if [ $nf -ne $nu ]; then
    echo "$0: ERROR: only $nf of $nu utterances produced features in $d."
    echo "$0: grep -i 'failed to process' $logdir/make_ssl_layers_${name}.*.log"
    $allow_missing || exit 1
    echo "$0: --allow-missing was set; continuing with $nf utterances."
  fi
  echo "$0: layer $L -> $d ($nf utterances)"
done

rm -f $logdir/wav_${name}.*.scp $logdir/utt2num_frames.* $logdir/utt2dur.*
echo "$0: Succeeded creating features for $name, layers $layers"
