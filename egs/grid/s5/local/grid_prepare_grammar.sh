#!/usr/bin/env bash

# Copyright 2025  (author: Ibrahim Almajai)   
# Apache 2.0.
echo "Preparing grammar for test"
[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

lang=data/lang

# Compile the grammar
compile () {
  cat $1 | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
          --keep_isymbols=false --keep_osymbols=false | fstarcsort \
          --sort_type=ilabel   > $2
}
 
# Create FST grammar for the GRID
echo "preparing fixed grammar"
local/create_chime1_grammar.pl > $lang/G.txt
compile $lang/G.txt $lang/G.fst

echo "preparing unigram grammar"
rm -rf ${lang}_ug
cp -r $lang ${lang}_ug
cut -d' ' -f2- data/train/text > txt
utils/make_unigram_grammar.pl <txt> ${lang}_ug/G.txt
compile ${lang}_ug/G.txt ${lang}_ug/G.fst
rm txt
 
# Draw the FST
#echo "fstdraw --isymbols=$lang/words.txt --osymbols=$lang/words.txt $lang/G.fst | dot -Tpdf > local/G.pdf"

exit 0
