#!/usr/bin/env bash

# Copyright 2013  (Author: Daniel Povey)
# Modified by Ibrahim Almjai (2025) 
# Apache 2.0

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
lmdir=data/local/kenlm
lexicon=data/local/dict/lexicon.txt
mkdir -p  $lmdir


# KenLM's lmplz builds the phone LMs. Set KENLM to the build directory (the one
# containing bin/lmplz) to use a build the search below does not find.
if [ -z "${KENLM:-}" ]; then
  for d in kenlm/build "$KALDI_ROOT/tools/kenlm/build" /data/git/kenlm/build ; do
    if [ -x "$d/bin/lmplz" ] ; then KENLM=$d ; break ; fi
  done
fi
[ -n "${KENLM:-}" ] && export PATH=${PATH}:$KENLM/bin

# lmplz may also already be on PATH, e.g. from a conda kenlm package.
if ! command -v lmplz >/dev/null ; then
  echo "===================================================================="
  echo "ERROR: KenLM's lmplz not found."
  echo "Searched: \$KENLM, ./kenlm/build, \$KALDI_ROOT/tools/kenlm/build,"
  echo "          /data/git/kenlm/build, and \$PATH."
  echo "To install KenLM:"
  echo "  git clone https://github.com/kpu/kenlm.git kenlm"
  echo "  cd kenlm"
  echo "  mkdir -p build"
  echo "  cd build"
  echo "  cmake .."
  echo "  make -j \$(nproc)"
  echo "Then rerun, or set KENLM=/path/to/kenlm/build if you built it elsewhere."
  exit 1
fi


# Create phone bigram and trigram LMs

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
