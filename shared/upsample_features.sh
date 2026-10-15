#!/usr/bin/env bash
# Copyright   2025  (author: Ibrahim Almajai)         
# Apache 2.0

nj=2
cmd=run.pl

interp_mode=linear
upsample_factor=2
compress=true    # false keeps float32; compression showed no WER effect on GRID and is ~4x smaller

echo "$0 $@"  

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 2 ] || [ $# -gt 4 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <src-data-dir> <dst-data-dir> [<log-dir> [<feats-dir>]]
 e.g.: $0 data/train data/train_upsampled
Note: <log-dir> defaults to <dst-data-dir>/log, and
      <feats-dir> defaults to <dst-data-dir>/data.
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
python shared/interpolate.py  --feats_scp=$logdir/feats_${name}.JOB.scp \
    --interp_mode=$interp_mode --upsample_factor=$upsample_factor ark:- \| \
    copy-feats --compress=$compress ark:- \
    ark,scp:$feats_dir/feats_$name.JOB.ark,$feats_dir/feats_$name.JOB.scp \
    || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $feats_dir/feats_$name.$n.scp || exit 1
done > $dst_data/feats.scp || exit 1

# run.pl gives copy-feats' status, which is 0 even when interpolate.py died
# partway, so compare counts rather than trust the exit code.
n_in=$(wc -l < $scp); n_out=$(wc -l < $dst_data/feats.scp)
if [ "$n_in" -ne "$n_out" ]; then
  echo "$0: ERROR: upsampled $n_out of $n_in utterances; see $feats_dir/make_interpolate_${name}.*.log"
  exit 1
fi
# Frame counts copied from the source are now half the real length.
rm -f $dst_data/utt2num_frames

if [ -f $src_data/frame_shift ]; then
  src_frame_shift=$(cat $src_data/frame_shift)
else
  echo "$0: WARNING: $src_data/frame_shift not found, assuming 0.02"
  src_frame_shift=0.02
fi
frame_shift=$(perl -e "print $src_frame_shift / $upsample_factor")
echo ${frame_shift} > $dst_data/frame_shift

echo "$0: Succeeded creating upsampled features for $name"

exit 0;
