#!/usr/bin/env bash
# Chain seeds 1 and 2 for frozen AV-HuBERT large L16 without CMVN (2026-10-17;
# seed 0 is e2e_tdnnf_frzlarge16norelu, 6.15 GRID / 4.89 Lombard, the paper's
# non-adapted figure). Same recipe as run_methods_queue.sh's nocmvn arm: only
# the chain seed changes. The training archives were deleted in a cleanup, so
# they are re-extracted first and dropped again at the end.
# Run from egs/grid/s5 in the ssl-kaldi environment:
#   ./run_frzlarge16_nocmvn_seeds.sh [--store DIR]
set -uo pipefail
store=   # feature store; default data/ft_store
. ./cmd.sh; . ./path.sh
. utils/parse_options.sh
export STORE=${store:-$PWD/data/ft_store}
tag=frzlarge16; off="--norm-means=false --norm-vars=false"

# A dir whose archives are gone would make run_ft_feats_arm.sh skip extraction.
up=data/train_${tag}_raw_upsampled
ark=$(head -1 $up/feats.scp 2>/dev/null | awk '{sub(/:[0-9]+$/,"",$2); print $2}')
[ -n "$ark" ] && [ ! -f "$ark" ] && rm -rf data/train_${tag}_raw $up

chains=""
for seed in 1 2; do
  c=e2e_tdnnf_${tag}norelu_s$seed; chains="$chains $c:$tag:exp/chain/e2e_tree_ft/graph"
  [ -f exp/chain/$c/final.mdl ] && continue
  echo "$(date +%H:%M) training $c"
  AFFIX=${tag}norelu_s$seed SEED=$seed CMVN_OPTS="$off" FEATDIM=1024 \
    ./run_ft_feats_arm.sh 16 input/large_vox_iter5.pt $tag > logs_frzl16nc_s$seed.log 2>&1 \
    || { echo "FAILED chain $c"; exit 1; }
  echo "$(date +%H:%M) done $c: $(grep SUMMARY logs_frzl16nc_s$seed.log)"
done
export EXTRA_CHAINS="$chains"
ONLY_TAGS=$tag STAGES="2" SETS=test ./run_rescore_r4000_lomb25.sh >> logs_frzl16nc_decode.log 2>&1 \
  || { echo "FAILED r4000 decode"; exit 1; }
ONLY_TAGS=$tag STAGES="1 2" ./run_shared_lombard.sh >> logs_frzl16nc_decode.log 2>&1 \
  || { echo "FAILED Lombard decode"; exit 1; }
rm -rf $STORE/ssl-kaldi-grid-$tag/train && echo "$(date +%H:%M) dropped train feats of $tag"

for seed in 0 1 2; do
  c=exp/chain/e2e_tdnnf_${tag}norelu$([ $seed = 0 ] || echo _s$seed)
  pooled=$(local/score_lombard_pooled.sh $c/decode_s_lomb_l $c/decode_s_lomb_p)
  echo "seed $seed: GRID $(grep WER $c/decode_r_test/wer_${pooled##*setting=} | awk '{print $2}')" \
    "| Lombard $(local/score_lombard_split.sh $c/decode_s_lomb_l data/lang/words.txt data/s_lomb_l_${tag}_raw)" \
    "| plain $(local/score_lombard_split.sh $c/decode_s_lomb_p data/lang/words.txt data/s_lomb_p_${tag}_raw)" \
    "| pooled $pooled"
done
echo FRZL16-NOCMVN-SEEDS-DONE
