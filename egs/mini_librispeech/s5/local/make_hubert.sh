#!/usr/bin/env bash
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0

nj=4
cmd=run.pl
compress=true
write_utt2num_frames=true  # If true writes utt2num_frames.
write_utt2dur=true
layer=9
model_type="facebook/hubert-base-ls960"

echo "$0 $@"  # Print the command line for logging.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
  cat >&2 <<EOF
Usage: $0 [options] <data-dir>  ]
 e.g.: $0 data/train
EOF
   exit 1;
fi

data=$1
logdir=$data/log
ssldir=$data/data

# make $ssldir an absolute pathname.
ssldir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $ssldir ${PWD}`

# use "name" as part of name of the archive.
name=`basename $data`

mkdir -p $ssldir || exit 1;
mkdir -p $logdir || exit 1;

if [ -f $data/feats.scp ]; then
  mkdir -p $data/.backup
  echo "$0: moving $data/feats.scp to $data/.backup"
  mv $data/feats.scp $data/.backup
fi

scp=$data/wav.scp

required="$scp"

for f in $required; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f"
    exit 1;
  fi
done

utils/validate_data_dir.sh --no-text --no-feats $data || exit 1;

for n in $(seq $nj); do
  utils/create_data_link.pl $ssldir/raw_hubert_$name.$n.ark
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

if [ -f $data/segments ]; then
  echo "$0 [info]: segments file exists: using that."

  split_segments=
  for n in $(seq $nj); do
    split_segments="$split_segments $logdir/segments.$n"
  done

  utils/split_scp.pl $data/segments $split_segments || exit 1;
  rm $logdir/.error 2>/dev/null

  $cmd JOB=1:$nj $logdir/make_hubert_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    python local/compute_hubert_feats.py --layer $layer \
         --model-type $model_type $write_utt2dur_opt ark:- ark:- \| \
         copy-feats --compress=$compress $write_num_frames_opt ark:- \
         ark,scp:$ssldir/raw_hubert_$name.JOB.ark,$ssldir/raw_hubert_$name.JOB.scp \
         || exit 1;
else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/make_hubert_${name}.JOB.log \
    python local/compute_hubert_feats.py --layer $layer \
          --model-type $model_type $write_utt2dur_opt \
	   scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
           copy-feats $write_num_frames_opt --compress=$compress ark:- \
      ark,scp:$ssldir/raw_hubert_$name.JOB.ark,$ssldir/raw_hubert_$name.JOB.scp \
      || exit 1;
fi

if [ -f $logdir/.error.$name ]; then
  echo "$0: Error producing features for $name:"
  tail $logdir/make_hubert_${name}.1.log
  exit 1;
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $ssldir/raw_hubert_$name.$n.scp || exit 1
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

frame_shift=0.02
echo $frame_shift > $data/frame_shift

rm $logdir/wav_${name}.*.scp  $logdir/segments.* \
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

echo "$0: Succeeded creating features for $name"
