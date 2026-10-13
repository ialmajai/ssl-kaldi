#!/usr/bin/env bash
# Rescore F03/M01 isolated conditions on the WavLM utterance set, at each arm's get_wer.py LMWT.
cd /data/learn/ssl-kaldi/egs/torgo/s5; . ./path.sh
ref_arm=wavlmbp
for a in mfcc n10 wavlmbp; do
  for sub in single single_nolimit; do
    lm=$(python3 local/get_wer.py exp/chain/e2e_tdnnf_$a 2>/dev/null | awk -v s="### $sub ###" '$0==s{f=1;next} f&&/Average LMWT/{printf "%d\n", $NF+0.5; exit}')
    for s in F03 M01; do
      d=exp/chain/e2e_tdnnf_$a/$s/decode_test_$sub
      keep=exp/chain/e2e_tdnnf_$ref_arm/$s/decode_test_$sub/scoring/test_filt.txt
      full=$(grep WER $d/wer_$lm | awk '{print $2}')
      inter=$(utils/filter_scp.pl $keep $d/scoring/$lm.tra | utils/int2sym.pl -f 2- exp/chain/e2e_tree_$s/graph_test_$sub/words.txt | sed 's:<UNK>::g' | \
        compute-wer --text --mode=present ark:$d/scoring/test_filt.txt ark,p:- 2>/dev/null | grep WER | awk '{print $2}')
      echo "$a $sub lmwt=$lm $s full=$full inter=$inter"
    done
  done
done
