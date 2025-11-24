#!/usr/bin/env bash
# Apache 2.0


srcdir=data/local/data
dir=data/local/dict
tmpdir=data/local/lm_tmp
mkdir -p $dir $tmpdir

[ -f path.sh ] && . ./path.sh

# silence phones, one per line.
echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# Create the lexicon, which is just an identity mapping
cut -d' ' -f2- data/train/text |  tr -cs '[:alnum:]\*\^\$\<' '[\n*]' | sort | uniq  > $dir/sws_arabic.txt
(cat  $dir/sws_arabic.txt ; echo "sil" ) | sort > $dir/phones.txt
paste $dir/phones.txt $dir/phones.txt > $dir/lexicon.txt || exit 1;
grep -v -F -f $dir/silence_phones.txt $dir/phones.txt > $dir/nonsilence_phones.txt 

# A few extra questions that will be added to those obtained by automatically clustering
# the "real" phones.  These ask about stress; there's also one for silence.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

echo "Dictionary & language model preparation succeeded"
