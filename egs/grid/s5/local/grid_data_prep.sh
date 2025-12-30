#!/bin/bash
#
# Copyright 2025 .  Apache 2.0.
# Modified by Ibrahim Almajai 2024
# To be run from one directory above this script.


if [ $# != 1 ]; then
  echo "Usage: local/data_prep.sh /path/to/dataset "
  exit 1; 
fi 

export LC_ALL=C

GRIDROOT=$1


tmpdir=data/local/tmp
mkdir -p $tmpdir
. ./path.sh || exit 1; 

dir=data/train
mkdir -p $dir

align_dir_to_text() {
  local align_root="$1"
  local out_text="$2"
  local exclude_speakers="${3:-}"   # e.g. "s1 s2 s20 s22"

  # Build the find arguments into an array
  local find_args=()

  if [ -n "$exclude_speakers" ]; then
    read -ra speakers <<< "$exclude_speakers"
    find_args+=( '(' )
    for sp in "${speakers[@]}"; do
      find_args+=( -name "$sp" -o )
    done
    # Remove the last -o
    unset 'find_args[-1]'
    find_args+=( ')' -prune -o )
  fi

  find_args+=( -name '*.align' -exec awk '
    {
      gsub(/\r$/, "")
      if (NF >= 3 && $3 != "sil" && $3 != "sp" && $3 != "") {
        if (words == "") words = $3
        else words = words " " $3
      }
    }
    ENDFILE {
      if (words != "") {
        gsub(/ $/, "", words)
        split(FILENAME, p, "/")
        speaker = p[length(p)-1]
        utt = p[length(p)]
        sub(/\.align$/, "", utt)
        print speaker "-" utt, words
      }
      words = ""
    }
  ' {} + )

  find "$align_root" "${find_args[@]}" | sort -k1 > "$out_text"
}


echo "Started train data prep. ..."

(
  find -L $GRIDROOT  -iname '*.mpg' | sed '/\/s1\//d' | sed '/\/s2\//d' | sed '/\/s20\//d' |\
	  sed '/\/s22\//d' | sed '/MACOSX/d'  | sort ;
) | perl -ane ' m:/sa\d.mpg:i || m:/sb\d\d.mpg:i || print; '  > $tmpdir/train.flist


#cat $tmpdir/train.flist | tr -s '//' '/'  > tmp
#mv tmp $tmpdir/train.flist

local/flist2scp.pl $tmpdir/train.flist | sort > $dir/video.scp


align_dir_to_text \
  "$GRIDROOT/alignments" \
  "$dir/text" \
  "s1 s2 s20 s22"

cat $dir/video.scp | perl -ane 'm/^((\w+)-\w+) / || die; print "$1 $2\n"' > $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt


echo "Started test data prep. ..."

dir=data/test
mkdir -p $dir

(
  find -L $GRIDROOT -iname '*.mpg' | sed -n -e '/\/s1\//p' -e '/\/s2\//p' -e '/\/s20\//p' -e '/\/s22\//p'\
	  |  sed '/MACOSX/d' | sort ;
) | perl -ane ' m:/sa\d.mpg:i || m:/sb\d\d.mpg:i || print; '  > $tmpdir/test.flist



local/flist2scp.pl $tmpdir/test.flist | sort > $dir/video.scp

align_dir_to_text \
  "$GRIDROOT/alignments" \
  "$dir/text" \
  "s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s23 s24 s25 s26 s27 s28 s29 s30 s31 s32 s33 s34"

sleep 0.25 

cat $dir/video.scp | perl -ane 'm/^((\w+)-\w+) / || die; print "$1 $2\n"' > $dir/utt2spk
utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt



