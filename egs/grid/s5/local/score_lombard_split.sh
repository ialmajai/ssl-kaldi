#!/usr/bin/env bash
# Score a Lombard decode separately on the dev speakers (local/lombard_dev_speakers.txt,
# 14 of 54, fixed seed) and the remaining eval speakers. Choices are made on dev and
# eval is reported (all 54 too, at the same setting), the LM weight and penalty included: for a Kaldi decode dir the
# setting with the lowest dev WER is applied unchanged to eval. A plain "utt words"
# hyp file is scored as is.
#   local/score_lombard_split.sh DECODE_DIR_OR_HYP [WORDS [LOMBARD_DATA_DIR]]
set -euo pipefail
. ./path.sh
src=$1; words=${2:-exp/chain/e2e_tree/graph/words.txt}
data=${3:-data/l4_lomb_raw}; ref=$data/text
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT

awk 'NR==FNR{d[$1]=1;next} {print $1, (($2 in d)?"dev":"eval")}' \
  local/lombard_dev_speakers.txt $data/utt2spk > $tmp/set
for s in dev eval; do awk -v s=$s '$2==s{print $1}' $tmp/set > $tmp/$s.list; done
cut -d' ' -f1 $tmp/set > $tmp/all.list

wer_on() {  # wer_on HYP SET
  utils/filter_scp.pl $tmp/$2.list $1 | compute-wer --text --mode=present ark:$ref ark:- 2>/dev/null \
    | awk '/WER/{print $2}'
}

if [ -f "$src" ]; then
  hyp=$src
else
  # Visit settings in full-set WER order so dev ties fall to the full-set best.
  best=; bestw=
  for f in $(grep -H WER $src/wer_* | sort -t' ' -k2 -g | cut -d: -f1); do
    lw=$(basename $f | sed 's/^wer_//'); lm=${lw%_*}; wip=${lw#*_}
    h=$src/scoring_kaldi/penalty_$wip/$lm.txt
    if [ ! -f $h ]; then
      h=$tmp/hyp_${lm}_$wip; utils/int2sym.pl -f 2- $words $src/scoring/$lm.$wip.tra > $h
    fi
    w=$(wer_on $h dev)
    if [ -z "$bestw" ] || awk -v a=$w -v b=$bestw 'BEGIN{exit !(a < b)}'; then best=$h; bestw=$w; fi
  done
  hyp=$best
fi
printf "dev=%s eval=%s all=%s\n" "$(wer_on $hyp dev)" "$(wer_on $hyp eval)" "$(wer_on $hyp all)"
