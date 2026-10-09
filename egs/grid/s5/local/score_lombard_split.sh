#!/usr/bin/env bash
# Score a Lombard decode separately on the dev speakers (local/lombard_dev_speakers.txt,
# 14 of 54, fixed seed) and the remaining eval speakers. Layer/system choices are
# made on dev; eval is reported. Takes a Kaldi decode dir (best LMWT/penalty from its
# full-set wer_* files, as bootci_pair.sh) or a plain "utt words" hyp file.
#   local/score_lombard_split.sh DECODE_DIR_OR_HYP [WORDS [LOMBARD_DATA_DIR]]
set -euo pipefail
. ./path.sh
src=$1; words=${2:-exp/chain/e2e_tree/graph/words.txt}
data=${3:-data/l4_lomb_raw}; ref=$data/text
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if [ -f "$src" ]; then hyp=$src
else
  # Read all of sort's output: head -1 under pipefail can kill sort with SIGPIPE (exit 141).
  best=$(grep -H WER $src/wer_* | sort -t' ' -k2 -g | awk -F: 'NR==1{print $1}')
  lw=$(basename $best | sed 's/^wer_//'); lm=${lw%_*}; wip=${lw#*_}
  if [ -f $src/scoring_kaldi/penalty_$wip/$lm.txt ]; then hyp=$src/scoring_kaldi/penalty_$wip/$lm.txt
  else utils/int2sym.pl -f 2- $words $src/scoring/$lm.$wip.tra > $tmp/hyp; hyp=$tmp/hyp; fi
fi
awk 'NR==FNR{d[$1]=1;next} {print $1, (($2 in d)?"dev":"eval")}' local/lombard_dev_speakers.txt $data/utt2spk > $tmp/set
for s in dev eval; do
  awk -v s=$s '$2==s{print $1}' $tmp/set > $tmp/$s.list
  w=$(utils/filter_scp.pl $tmp/$s.list $hyp | compute-wer --text --mode=present ark:$ref ark:- 2>/dev/null | grep WER | awk '{print $2}')
  printf "%s=%s " $s $w
done; echo
