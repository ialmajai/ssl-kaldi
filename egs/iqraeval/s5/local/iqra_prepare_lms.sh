#!/usr/bin/env bash

# Copyright 2013  (Author: Daniel Povey)
# Modified by Ibrahim Almjai (2025) 
# Apache 2.0

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
lmdir=data/local/kenlm
lexicon=data/local/dict/lexicon.txt
KENLM=/data/git/kenlm/build
mkdir -p  $lmdir


# Check for KenLM (used for faster/better ARPA LM training and querying in some recipes)
if [ ! -x $KENLM/bin/lmplz ] ; then
  echo "===================================================================="
  echo "WARNING: KenLM not found or not built in $KENLM"
  echo "To install KenLM:"
  echo "  git clone https://github.com/kpu/kenlm.git kenlm"
  echo "  cd kenlm"
  echo "  mkdir -p build"
  echo "  cd build"
  echo "  cmake .."
  echo "  make -j $(nproc)"
  echo "If successful update the above KENLM path accordingly"
  exit 1
fi


# Create phone bigram and trigram LMs
export PATH=${PATH}:$KALDI_ROOT/tools/kenlm/build/bin

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
