#!/usr/bin/env bash
# Seeds 1 and 2 of the mHuBERT-147 LoRA fine-tune (seed 0 is exp_ft/mhubert147_lora32,
# 19.07 through the chain), each run through run_ctc_finetune.sh stages 1-5:
# fine-tune with the checkpoint and LM weights chosen on the two held-out
# sessions, self-decodes, layer-9 features and the n10 e2e chain. Prints the
# chain WER at the test-best LM weight and at the cross-selected one.
# Run from egs/swahili/s5 in the ssl-kaldi environment:
#   ./run_ft_seeds_mhubert_lora.sh [--seeds "1 2"] [--store DIR]
set -uo pipefail
seeds="1 2"
store=   # fine-tuned-encoder features; default data/ft_store
. ./cmd.sh; . ./path.sh
. utils/parse_options.sh
store=${store:-$PWD/data/ft_store}/swahili-ft-lora32-seeds

for seed in $seeds; do
  dir=exp/chain/e2e_tdnnf_n10_ft_lora32_s$seed
  ls $dir/decode_ft_lora32_s$seed/wer_* >/dev/null 2>&1 && continue   # already decoded
  echo "######## $(date +%H:%M:%S) seed $seed"
  ./run_ctc_finetune.sh --arm lora32 --seed $seed --stage 1 --store $store \
    || { echo "FAILED seed $seed"; exit 1; }
done

echo "==== mHuBERT-147 LoRA r32, chain on fine-tuned L9 features, OpenSLR-25 test"
for seed in 0 $seeds; do
  sfx=$([ $seed = 0 ] || echo _s$seed)
  d=exp/chain/e2e_tdnnf_n10_ft_lora32$sfx/decode_ft_lora32$sfx
  echo "  seed $seed: self-decode $(grep -h 'WER' exp_ft/mhubert147_lora32$sfx/lm_test.log 2>/dev/null | tail -1)"
  python shared/score_cross_select.py $d data/test/utt2spk --words exp/chain/e2e_tree/graph/words.txt | sed 's/^/    /'
done
echo "MHUBERT-LORA-SEEDS-DONE"
