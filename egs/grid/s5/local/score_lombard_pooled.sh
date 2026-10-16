#!/usr/bin/env bash
# Score a system on all 5,142 shared Lombard GRID clips (Lombard + plain speech,
# 30,852 words), pooling its two decodes. The LM weight and penalty are chosen
# on the 14 dev speakers over the pooled clips, then applied unchanged to the
# 40 eval speakers and to all 54.
#   local/score_lombard_pooled.sh DECODE_DIR_LOMBARD DECODE_DIR_PLAIN [WORDS]
set -euo pipefail
. ./path.sh
L=$1; P=$2; words=${3:-data/lang/words.txt}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
cat data/s_lomb_l_l4_raw/text data/s_lomb_p_l4_raw/text > $tmp/ref
cat data/s_lomb_l_l4_raw/utt2spk data/s_lomb_p_l4_raw/utt2spk > $tmp/u2s
awk 'NR==FNR{d[$1]=1;next} {print $1, (($2 in d)?"dev":"eval")}' \
  local/lombard_dev_speakers.txt $tmp/u2s > $tmp/set
for s in dev eval; do awk -v s=$s '$2==s{print $1}' $tmp/set > $tmp/$s.list; done
cut -d' ' -f1 $tmp/set > $tmp/all.list
hyp_of() {  # hyp_of DECODE_DIR LMWT WIP
  local h=$1/scoring_kaldi/penalty_$3/$2.txt
  if [ -f $h ]; then cat $h; else utils/int2sym.pl -f 2- $words $1/scoring/$2.$3.tra; fi
}
wer_on() { utils/filter_scp.pl $tmp/$2.list $1 | compute-wer --text --mode=present ark:$tmp/ref ark:- 2>/dev/null | awk '/WER/{print $2}'; }
best=; bestw=
for f in $(ls $L/wer_* | sort); do
  lw=$(basename $f | sed 's/^wer_//'); lm=${lw%_*}; wip=${lw#*_}
  [ -f $P/wer_$lw ] || continue
  { hyp_of $L $lm $wip; hyp_of $P $lm $wip; } > $tmp/h_$lw
  w=$(wer_on $tmp/h_$lw dev)
  if [ -z "$bestw" ] || awk -v a=$w -v b=$bestw 'BEGIN{exit !(a<b)}'; then best=$lw; bestw=$w; fi
done
h=$tmp/h_$best
printf "dev=%s eval=%s all=%s words=%s setting=%s\n" "$(wer_on $h dev)" "$(wer_on $h eval)" \
  "$(wer_on $h all)" "$(utils/filter_scp.pl $tmp/all.list $tmp/ref | awk '{n+=NF-1} END{print n}')" $best
