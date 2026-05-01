#!/usr/bin/env bash
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

nj=2
cmd=run.pl

#interpolation
interp_mode=
upsample_factor=2

echo "$0 $@"  

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

src_data=$1
dst_data=$2

if [ $# -ge 3 ]; then
  logdir=$3
else
  logdir=$dst_data/log
fi
if [ $# -ge 4 ]; then
  feats_dir=$4
else
  feats_dir=$dst_data/data
fi

# use "name" as part of name of the archive.
name=`basename $src_data`

if [ -f $dst_data/feats.scp ]; then
  mkdir -p $dst_data/.backup
  echo "$0: moving $dst_data/feats.scp to $dst_data/.backup"
  mv $dst_data/feats.scp $dst_data/.backup
fi

mkdir -p $feats_dir || exit 1;
mkdir -p $logdir || exit 1;

split_scps=
for n in $(seq $nj); do
    split_scps="$split_scps $logdir/feats_${name}.$n.scp"
done

scp=$src_data/feats.scp

utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $feats_dir/make_interpolate_${name}.JOB.log \
python local/interpolate.py  --feats_scp=$logdir/feats_${name}.JOB.scp \
    --interp_mode=$interp_mode --upsample_factor=$upsample_factor ark:- \| \
    copy-feats ark:- \
    ark,scp:$feats_dir/feats_$name.JOB.ark,$feats_dir/feats_$name.JOB.scp \
    || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $feats_dir/feats_$name.$n.scp || exit 1
done > $dst_data/feats.scp || exit 1

frame_shift=0.02
if [ $upsample_factor -eq 4 ]; then
    frame_shift=0.01 
fi
echo ${frame_shift} > $dst_data/frame_shift

echo "$0: Succeeded creating PCA features for $name"

exit 0;
