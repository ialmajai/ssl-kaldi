#!/usr/bin/env bash
# Like make_avhubert.sh, but for building an *augmented* copy of a training
# set for multi-condition training: passes --augment/--augment-seed through
# to compute_avhubert_feats.py (flip/rotate/crop-jitter/blur/noise on the
# ROI before AV-HuBERT extraction, see augment_roi_frames's docstring for
# why this is offline rather than a real on-the-fly augmenter -- Kaldi's
# chain training has no per-epoch hook to run one from). Also fixed to
# meanface mode with --mean-face/--window-margin passthrough, which the
# original make_avhubert.sh predates and doesn't expose.
#
# <data-dir>'s video.scp should already have augmented-copy utt-ids (e.g.
# an "aug-" prefix) distinct from the clean copy's, so the two can be
# combined into one training set afterwards without utt-id collisions --
# this script does not do that renaming itself.

nj=2
cmd=run.pl
compress=true
write_utt2num_frames=true
write_utt2dur=true
layer=9
ckpt=input/base_vox_iter5.pt
avhubert_path=av_hubert
mean_face=av_hubert/avhubert/20words_mean_face.npy
window_margin=12
min_frames=74
augment_seed=0

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <data-dir> [<log-dir> [<avhubert-dir>] ]
 e.g.: $0 --nj 6 --layer 4 --augment-seed 42 data/train_l4_raw_aug
EOF
   exit 1;
fi

data=$1
if [ $# -ge 2 ]; then
  logdir=$2
else
  logdir=$data/log
fi
if [ $# -ge 3 ]; then
  ssldir=$3
else
  ssldir=$data/data
fi

ssldir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ssldir ${PWD}`

name=`basename $data`

mkdir -p $ssldir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/video.scp

required="$scp $ckpt $mean_face"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

if [ ! -d $avhubert_path ]; then
  echo "$0: av_hubert codebase not found at $avhubert_path;"
  echo "    clone it as described in the README or pass --avhubert-path"
  exit 1;
fi

for n in $(seq $nj); do
  utils/create_data_link.pl $ssldir/raw_avhubert_$name.$n.ark
done

if $write_utt2num_frames; then
  write_num_frames_opt="--write-num-frames=ark,t:$logdir/utt2num_frames.JOB"
else
  write_num_frames_opt=
fi

if $write_utt2dur; then
  write_utt2dur_opt="--write-utt2dur ark,t:$logdir/utt2dur.JOB"
else
  write_utt2dur_opt=
fi

split_scps=
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/video_${name}.$n.scp"
done

utils/split_scp.pl $scp $split_scps || exit 1;

$cmd JOB=1:$nj $logdir/make_avhubert_${name}.JOB.log \
  python local/compute_avhubert_feats.py --layer $layer --crop-mode meanface \
    --mean-face $mean_face --window-margin $window_margin --min-frames $min_frames \
    --augment --augment-seed $augment_seed \
    $write_utt2dur_opt \
    --ckpt $ckpt --path $avhubert_path scp,p:$logdir/video_${name}.JOB.scp ark:- \| \
    copy-feats $write_num_frames_opt --compress=$compress ark:- \
    ark,scp:$ssldir/raw_avhubert_$name.JOB.ark,$ssldir/raw_avhubert_$name.JOB.scp \
    || exit 1;

for n in $(seq $nj); do
  cat $ssldir/raw_avhubert_$name.$n.scp || exit 1
done > $data/feats.scp || exit 1

if $write_utt2num_frames; then
  for n in $(seq $nj); do
    cat $logdir/utt2num_frames.$n || exit 1
  done > $data/utt2num_frames || exit 1
fi

if $write_utt2dur; then
  for n in $(seq $nj); do
    cat $logdir/utt2dur.$n || exit 1
  done > $data/utt2dur || exit 1
fi

frame_shift=0.04
echo ${frame_shift} > $data/frame_shift

rm $logdir/video_${name}.*.scp \
   $logdir/utt2num_frames.* $logdir/utt2dur.* 2>/dev/null

nf=$(wc -l < $data/feats.scp)
nu=$(wc -l < $data/utt2spk)
if [ $nf -ne $nu ]; then
  echo "$0: It seems not all of the feature files were successfully procesed" \
       "($nf != $nu); consider using utils/fix_data_dir.sh $data"
fi

if (( nf < nu - nu/20 )); then
  echo "$0: Less than 95% the features were successfully generated."\
       "Probably a serious error."
  exit 1
fi

echo "$0: Succeeded creating augmented AV-HuBert features for $name"
