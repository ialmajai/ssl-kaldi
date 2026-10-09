#!/usr/bin/env bash
# Word confidence for the LoRA base L10 aug2x demo model: Kaldi logistic
# calibration on the GRID test lattices (as the aug2x demo model), then the
# TruCLeS CEM and its late fusion with MBR on the same splits as exp/trucles/aug2x.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/data/miniconda3/envs/ssl-kaldi/bin:$PATH
. ./cmd.sh; . ./path.sh
chain=exp/chain/e2e_tdnnf_lor10norelu_cmvn_aug2x; out=exp/trucles/base_lor10_aug2x
step() { echo "######## $(date +%H:%M:%S) [base-lor10-conf] $*"; }

step "Kaldi calibration"
steps/conf/train_calibration.sh --cmd "$decode_cmd" --lmwt 10 \
  data/test_lor10_raw_upsampled data/lang data/lang/word_feats $chain/decode $chain/calibration

for s in test:test_lor10_raw_upsampled:decode lomb:lor10_lomb_raw_upsampled:decode_lombard; do
  IFS=: read name data dec <<< "$s"; o=$out/$name
  step "CEM data $name"
  local/demo/trucles/make_cem_data.sh $chain data/$data $chain/$dec $o
  ln -sfn $PWD/data/$data $o/data
  lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c $chain/$dec/lat.*.gz|" ark:- | \
    lattice-align-words-lexicon data/lang/phones/align_lexicon.int $chain/final.mdl ark:- ark:- | \
    lattice-to-ctm-conf --decode-mbr=true --frame-shift=1 ark:- - 2>/dev/null | \
    utils/int2sym.pl -f 5 data/lang/words.txt > $o/mbr.ctm
done
rm -f $out/*/raw.nnet $out/*/hidden.nnet

tr="$out/test:~s1,s2 $out/lomb:+local/lombard_dev_speakers.txt"
ev="$out/test:~s20,s22 $out/lomb:-local/lombard_dev_speakers.txt"
step "CEM"
python3 local/demo/trucles/train_cem.py --train $tr --eval $ev --out $out/full | tee $out/full.log
step "fusion"
python3 local/demo/trucles/fuse_mbr_cem.py --train $tr --eval $ev --temp $(awk '/^using temp/{print $3}' $out/full.log) --out $out/fusion
# Write the net the demo's fusion runs: one pass giving [prefinal activations, logits].
nnet3-am-copy --raw=true $chain/final.mdl - | nnet3-copy --nnet-config=<(echo \
  "output-node name=output input=Append(prefinal-chain.batchnorm2, output.affine)") - $out/fusion/cem.raw
echo "DEMO-LOR10-CONF-DONE"
