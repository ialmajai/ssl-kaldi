#!/usr/bin/env bash
# Fine-tune seeds 43 and 44 for the large full fine-tune (2026-10-17; the
# original, seed 1337, is e2e_tdnnf_lgf20norelu_cmvn, the paper's single-run
# large full row). Each encoder is read at L20 through the same e2e chain
# (chain seed 0, so only the fine-tune varies) and decoded on the 4,000-clip
# GRID test and the shared Lombard set. Large counterpart of run_ft_seeds_base.sh.
# Run from egs/grid/s5 in the ssl-kaldi environment:
#   ./run_ft_seeds_large_full.sh [--store DIR]
set -uo pipefail
store=   # feature store; default data/ft_store
. ./cmd.sh; . ./path.sh
. utils/parse_options.sh
export STORE=${store:-$PWD/data/ft_store}
need_space() {
  local d free
  for d in $PWD $STORE; do
    mkdir -p $d; free=$(df --output=avail -BG $d | tail -1 | tr -dc 0-9)
    [ $free -ge 12 ] || { echo "$(date +%H:%M) only ${free}G free on $d, stopping before $1"; exit 1; }
  done
}

for s in 43 44; do
  cfg=large_ctc_full_s$s; tag=lgf20s$s; ckpt=input/avhubert_$cfg.pt
  if [ ! -f $ckpt ]; then
    need_space $cfg
    echo "$(date +%H:%M) fine-tuning $cfg"
    ./run_ctc_finetune.sh --config $cfg --ft-dir finetune_lora --stage 2 --stop-stage 3 \
      > logs_ftseeds_$cfg.log 2>&1 || { echo "FAILED fine-tune $cfg"; exit 1; }
    [ -f $ckpt ] || { echo "FAILED export $cfg"; exit 1; }
    grep -o '"valid_wer": "[0-9.]*"' finetune_lora/exp_$cfg/hydra_train.log | sort -t'"' -k4 -g | head -1
    rm -rf finetune_lora/exp_$cfg/checkpoints
  fi
  c=e2e_tdnnf_${tag}norelu_cmvn
  if [ ! -f exp/chain/$c/final.mdl ]; then
    need_space $tag
    hostname > exp/chain/.claim_$tag
    FEATDIM=1024 ./run_ft_feats_arm.sh 20 $ckpt $tag > logs_ftseeds_$tag.log 2>&1 \
      || { echo "FAILED chain $tag"; exit 1; }
    echo "$(date +%H:%M) done $tag: $(grep SUMMARY logs_ftseeds_$tag.log)"
  fi
  export EXTRA_CONFIGS="$tag:$ckpt:20"
  export EXTRA_CHAINS="$c:$tag:exp/chain/e2e_tree_ft/graph"
  ONLY_TAGS=$tag STAGES="1 2" SETS=test ./run_rescore_r4000_lomb25.sh >> logs_ftseeds_large_decode.log 2>&1 \
    || { echo "FAILED r4000 decode $tag"; exit 1; }
  ONLY_TAGS=$tag STAGES="1 2" ./run_shared_lombard.sh >> logs_ftseeds_large_decode.log 2>&1 \
    || { echo "FAILED Lombard decode $tag"; exit 1; }
  rm -rf $STORE/ssl-kaldi-grid-$tag/train && echo "$(date +%H:%M) dropped train feats of $tag"
done

for tag in lgf20 lgf20s43 lgf20s44; do
  c=exp/chain/e2e_tdnnf_${tag}norelu_cmvn
  pooled=$(local/score_lombard_pooled.sh $c/decode_s_lomb_l $c/decode_s_lomb_p)
  echo "$tag: GRID $(grep WER $c/decode_r_test/wer_${pooled##*setting=} | awk '{print $2}')" \
    "| Lombard $(local/score_lombard_split.sh $c/decode_s_lomb_l data/lang/words.txt data/s_lomb_l_${tag}_raw)" \
    "| plain $(local/score_lombard_split.sh $c/decode_s_lomb_p data/lang/words.txt data/s_lomb_p_${tag}_raw)" \
    "| pooled $pooled"
done
echo FT-SEEDS-LARGE-FULL-DONE
