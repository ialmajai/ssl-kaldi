#!/usr/bin/env bash
# Dump what the TruCLeS confidence model needs for one decoded set: 1-best word
# frame spans, per-frame phone posteriors, and per-frame net activations
# (prefinal hidden, output logits), all at the lattice frame rate. The input
# features are read from DATA_DIR by train_cem.py.
#   local/demo/trucles/make_cem_data.sh CHAIN_DIR DATA_DIR DECODE_DIR OUT_DIR [LMWT]
set -euo pipefail
cd "$(dirname "$0")/../../.."   # egs/grid/s5
. ./cmd.sh; . ./path.sh
chain=$1 data=$2 dec=$3 out=$4 lmwt=${5:-10}
lex=data/lang/phones/align_lexicon.int
mkdir -p $out
fsf=$(cat $chain/frame_subsampling_factor)
lat="ark:gunzip -c $dec/lat.*.gz|"

# Read 1-best word spans; frame-shift 1 makes the ctm columns lattice frame indices.
lattice-scale --inv-acoustic-scale=$lmwt "$lat" ark:- | lattice-1best ark:- ark:- | \
  lattice-align-words-lexicon $lex $chain/final.mdl ark:- ark:- | \
  nbest-to-ctm --frame-shift=1 --print-silence=false ark:- - | \
  utils/int2sym.pl -f 5 data/lang/words.txt > $out/words.ctm

# Write the 1-best per-frame phones and the pdf -> phone map (true-class
# probabilities come from the net's softmax summed per phone).
lattice-scale --inv-acoustic-scale=$lmwt "$lat" ark:- | lattice-1best ark:- ark:- | \
  nbest-to-linear ark:- ark:- | ali-to-phones --per-frame=true $chain/final.mdl ark:- ark,t:$out/phones.txt
show-transitions data/lang/phones.txt $chain/final.mdl | \
  awk '/Transition-state/{for(i=1;i<=NF;i++){if($i=="phone")ph=$(i+2); if($i~/-pdf$/)print $(i+2), ph}}' | \
  sort -n -u > $out/pdf2phone.txt

# Run the net on utterance-CMVN features, as the demo does; a copy whose
# output node is the prefinal layer gives the hidden activations.
nnet3-am-copy --raw=true $chain/final.mdl $out/raw.nnet
nnet3-copy --nnet-config=<(echo "output-node name=output input=prefinal-chain.batchnorm2") \
  $out/raw.nnet $out/hidden.nnet
feats="ark:apply-cmvn --norm-vars=true 'ark:compute-cmvn-stats scp:$data/feats.scp ark:- |' scp:$data/feats.scp ark:- |"
opts="--frame-subsampling-factor=$fsf --frames-per-chunk=150 --extra-left-context-initial=0 --extra-right-context-final=0 --use-gpu=no"
nnet3-compute $opts $out/raw.nnet "$feats" ark:$out/logits.ark
nnet3-compute $opts $out/hidden.nnet "$feats" ark:$out/hidden.ark
cp $data/text $out/text
echo "make_cem_data: $(wc -l < $out/words.ctm) words -> $out"
