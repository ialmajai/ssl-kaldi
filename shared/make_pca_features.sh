#!/usr/bin/env bash
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

nj=2
cmd=run.pl

pca_dim=30
pca_model="ipca.pt"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

echo "$0 $@"

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <src-data-dir> <dst-data-dir> [<log-dir> [<pca-dir>]]
 e.g.: $0 data/train_raw data/train
Note: <log-dir> defaults to <dst-data-dir>/log, and
      <pca-dir> defaults to <dst-data-dir>/data.
EOF
  exit 1;
fi

src_data=$1
dst_data=$2

if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=$dst_data/log
fi
if [ $# -ge 4 ]; then
  pcadir=$4
else
  pcadir=$dst_data/data
fi

# use "name" as part of name of the archive.
name=`basename $src_data`

if [ -f $dst_data/feats.scp ]; then
  mkdir -p $dst_data/.backup
  echo "$0: moving $dst_data/feats.scp to $dst_data/.backup"
  mv $dst_data/feats.scp $dst_data/.backup
fi

mkdir -p $pcadir || exit 1;
mkdir -p $logdir || exit 1;
split_scps=
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/feats_${name}.$n.scp"
done

scp=$src_data/feats.scp

utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $logdir/make_pca_${name}.JOB.log \
python shared/pca.py  --pca_dim=$pca_dim --mode=apply \
    --feats_scp=$logdir/feats_${name}.JOB.scp \
    --pca_model=$pca_model ark:- \
    \|  copy-feats ark:- \
    ark,scp:$pcadir/feats_$name.JOB.ark,$pcadir/feats_$name.JOB.scp \
    || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $pcadir/feats_$name.$n.scp || exit 1
done > $dst_data/feats.scp || exit 1

# PCA does not change the frame rate.
if [ -f $src_data/frame_shift ]; then
  cp $src_data/frame_shift $dst_data/frame_shift
fi

nf=$(wc -l < $dst_data/feats.scp)
nu=$(wc -l < $src_data/feats.scp)
if [ $nf -ne $nu ]; then
  echo "$0: It seems not all of the feature files were successfully processed" \
       "($nf != $nu); consider using utils/fix_data_dir.sh $dst_data"
  exit 1
fi

echo "$0: Succeeded creating PCA features for $name"
