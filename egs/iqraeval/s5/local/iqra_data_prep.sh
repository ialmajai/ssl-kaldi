#!/bin/bash

# Copyright 2025, author: Ibrahim Almajai         
# Apache 2.0

if [ $# != 1 ]; then
  echo "Usage: local/iqra_data_prep.sh /path/to/IqraEval-dataset "
  exit 1; 
fi 

export LC_ALL=C

IQRAROOT=$1

tmpdir=data/local/tmp
mkdir -p $tmpdir
. ./path.sh || exit 1; 

label_convert() {
    local label_dir="$1"      

    find $label_dir -name 'transcript_*.txt' -print0 | \
       xargs -0 -I{} awk '
        BEGIN {OFS=" "}
        {
          split(FILENAME, a, "/"); fname=a[length(a)]
          utt_id=substr(fname, 1, length(fname)-4)
	  # Convert to lowercase and replace underscores
          utt_id = tolower(utt_id)
          gsub(/_/, "-", utt_id)
	  gsub(/\(/, "", utt_id)
	  gsub(/\)/, "", utt_id)
          printf "%s %s\n", utt_id, $0
        }
      ' '{}'
}

###################
#prepare train data
###################

dir=data/train
mkdir -p $dir

(
find $IQRAROOT/TTS/train   -iname '*.wav' | sort ;
) | perl -ane ' m:/sa\d.wav:i || m:/sb\d\d.wav:i || print; '  > $tmpdir/train_wav.flist

local/flist2scp.pl $tmpdir/train_wav.flist | sort  | sed 's:audio-:tts-:'   > $dir/wav.scp

(
find   $IQRAROOT/CV-Ar/train -iname '*.wav' | sort ;
) | perl -ane ' m:/sa\d.wav:i || m:/sb\d\d.wav:i || print; '  > $tmpdir/train_wav.flist

local/flist2scp.pl $tmpdir/train_wav.flist | sort | sed 's:audio-:cvar-:'  >> $dir/wav.scp

sed -i -e 's: : sox ":' -e 's:$:" -t wav -b 16 -e signed-integer -r 16000 - |:' $dir/wav.scp

rm -f  $dir/text

label_convert $IQRAROOT/TTS/train   | sed 's:transcript:tts:' > $dir/text
label_convert $IQRAROOT/CV-Ar/train | sed 's:transcript:cvar:' >> $dir/text


awk '{print $1,$1}' $dir/wav.scp  > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt

###################
#prepare dev data
###################

dir=data/dev
mkdir -p $dir

(
  find $IQRAROOT/CV-Ar/dev -iname '*.wav' | sort ;
) | perl -ane ' m:/sa\d.wav:i || m:/sb\d\d.wav:i || print; '  > $tmpdir/dev_wav.flist

local/flist2scp.pl $tmpdir/dev_wav.flist | sort  | sed 's:audio-:cvar-:'   > $dir/wav.scp

sed -i -e 's: : sox ":' -e 's:$:" -t wav -b 16 -e signed-integer -r 16000 - |:' $dir/wav.scp

rm -f  $dir/text
label_convert $IQRAROOT/CV-Ar/dev  | sed 's:transcript:cvar:' > $dir/text

sort -k1 -o $dir/text $dir/text

awk '{print $1,$1}' $dir/wav.scp  > $dir/utt2spk
cp $dir/utt2spk $dir/spk2utt
