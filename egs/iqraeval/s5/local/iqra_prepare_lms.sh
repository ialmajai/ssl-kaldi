#!/usr/bin/env bash

# Copyright 2013  (Author: Daniel Povey)
# Modified by Ibrahim Almjai (2025) 
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
lmdir=data/local/kenlm
lexicon=data/local/dict/lexicon.txt
KENLM=/data/git/kenlm/build
mkdir -p  $lmdir

# Create phone bigram and trigram LMs
if [ -z $KENLM ] ; then
  export IRSTLM=$KALDI_ROOT/tools/irstlm/
fi
export PATH=${PATH}:$KENLM/bin
if ! command -v lmplz >/dev/null 2>&1 ; then
  echo "$0: Error: kenlm is not available or compiled" >&2
  exit 1
fi

cut -d' ' -f2- data/train/text | sort | uniq  > $lmdir/lm_train.text

lmplz -o 2 --discount_fallback <  $lmdir/lm_train.text > $lmdir/bigram.lm.arpa
lmplz -o 3 --discount_fallback <  $lmdir/lm_train.text > $lmdir/trigram.lm.arpa


for lm in bigram trigram ; do
  test=data/lang_test_${lm}
  mkdir -p $test
  cp -r data/lang/* $test

  cat $lmdir/${lm}.lm.arpa | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - $test/G.fst
  fstisstochastic $test/G.fst
  utils/validate_lang.pl data/lang_test_${lm} || exit 1
done

echo "Succeeded in LMs prep."
