#!/usr/bin/env bash
# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0

nj=4
cmd=run.pl
compress=true
write_utt2num_frames=true  # If true writes utt2num_frames.
write_utt2dur=true
layer=9
ssl_model="facebook/hubert-base-ls960"
allow_missing=false  # if true, tolerate utterances that produced no features
max_failures=0       # how many utterances may fail before a job aborts
# keep|strip: see compute_ssl_feats.py. Only bites on models that apply the
# encoder-final layer_norm after the layer loop (the large ones, and Whisper).
# keep is the measured-better default; strip gives features that match
# hidden_states[layer] on the full model, for comparability with the literature.
final_layer_norm=keep

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
  utils/create_data_link.pl $ssldir/raw_ssl_$name.$n.ark
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

  $cmd JOB=1:$nj $logdir/make_ssl_${name}.JOB.log \
    extract-segments scp,p:$scp $logdir/segments.JOB ark:- \| \
    python shared/compute_ssl_feats.py --layer $layer \
         --ssl-model $ssl_model --final-layer-norm $final_layer_norm \
         --max-failures $max_failures $write_utt2dur_opt ark:- ark:- \| \
         copy-feats --compress=$compress $write_num_frames_opt ark:- \
         ark,scp:$ssldir/raw_ssl_$name.JOB.ark,$ssldir/raw_ssl_$name.JOB.scp \
         || exit 1;
else
  echo "$0: [info]: no segments file exists: assuming wav.scp indexed by utterance."
  split_scps=
  for n in $(seq $nj); do
    split_scps="$split_scps $logdir/wav_${name}.$n.scp"
  done

  utils/split_scp.pl $scp $split_scps || exit 1;

  $cmd JOB=1:$nj $logdir/make_ssl_${name}.JOB.log \
    python shared/compute_ssl_feats.py --layer $layer \
          --ssl-model $ssl_model --final-layer-norm $final_layer_norm \
          --max-failures $max_failures $write_utt2dur_opt \
	   scp,p:$logdir/wav_${name}.JOB.scp ark:- \| \
           copy-feats $write_num_frames_opt --compress=$compress ark:- \
      ark,scp:$ssldir/raw_ssl_$name.JOB.ark,$ssldir/raw_ssl_$name.JOB.scp \
      || exit 1;
fi

# A job that aborted leaves no utt2dur, so without this the failure surfaces
# further down as a confusing "cat: utt2dur.1: No such file".
if grep -qi "^aborting after" $logdir/make_ssl_${name}.*.log 2>/dev/null; then
  echo "$0: ERROR: feature extraction aborted."
  grep -hiE "Failed to process|^aborting after" $logdir/make_ssl_${name}.*.log \
    | sed "s|^|$0:   |"
  echo "$0: If it is CUDA OOM, re-run with a smaller --nj (currently $nj)."
  echo "$0: To tolerate failures instead, re-run with a larger --max-failures"
  echo "$0: (currently $max_failures) and --allow-missing true."
  exit 1
fi

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $ssldir/raw_ssl_$name.$n.scp || exit 1
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
  # Unlike MFCCs, where a dropped utterance means a corrupt wav, here it is
  # usually a transient CUDA OOM: the utterance is fine and would succeed on a
  # quieter GPU. Tolerating it silently shrinks the data directory, and a test
  # set that lost utterances cannot be compared against a run that kept them.
  echo "$0: ERROR: only $nf of $nu utterances produced features."
  echo "$0: grep -i 'failed to process' $logdir/make_ssl_${name}.*.log"
  echo "$0: If it is CUDA OOM, re-run with a smaller --nj (currently $nj)."
  echo "$0: To accept the loss instead, pass --allow-missing true and then run"
  echo "$0: utils/fix_data_dir.sh $data"
  $allow_missing || exit 1
  # The 95% floor still applies even when losses are tolerated: --allow-missing
  # is for a handful of transient OOMs, not for silently accepting a data
  # directory that lost a twentieth of its utterances. Checked here rather than
  # unconditionally below so that the message matches what actually happens.
  if (( nf < nu - nu/20 )); then
    echo "$0: ERROR: that is more than 5% of the data, which --allow-missing"
    echo "$0: does not cover. Fix the cause instead."
    exit 1
  fi
  echo "$0: --allow-missing was set; continuing with $nf of $nu utterances."
fi

echo "$0: Succeeded creating features for $name"
