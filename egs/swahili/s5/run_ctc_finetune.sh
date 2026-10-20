#!/usr/bin/env bash
# Copyright   2026  Ibrahim Almajai
# License: Apache 2.0.
#
# Fine-tuned baseline for the frozen-feature chain: fine-tune mHuBERT-147 on
# the Swahili training set with a character CTC head, then
#   - decode it directly, greedy and with the corpus trigram (beam search), and
#   - use the fine-tuned encoder as the feature extractor for the same
#     renorm-linear 10-layer e2e chain as the frozen 19.47.
# Run after run_e2e.sh stages 0-2 (data/train, data/test, data/train_spe2e and
# the e2e tree from the frozen run).
#
#   ./run_ctc_finetune.sh                       # LoRA r32 on attention + FFN
#   ./run_ctc_finetune.sh --arm full            # every encoder weight trainable
#   ./run_ctc_finetune.sh --seed 1 --stage 1    # repeat run; dirs get _s1
# --store DIR puts the stage-4 features under DIR, linked from data/.
#
# Two training sessions (SWH-05-20110327, SWH-15-20110310; 668 utterances) are
# held out to choose the checkpoint and the LM weights, so the test set chooses
# nothing.
#
# Stages:
#   0  data/train_ctc and data/valid_ctc: text without <UNK>, <music>,
#      <laughter> and characters outside [a-z0-9'-]
#   1  fine-tune (one GPU)
#   2  greedy CTC decode of valid and test, no LM
#   3  beam search with the corpus trigram; alpha and beta chosen on valid
#   4  fine-tuned encoder features (layer 9) for train_spe2e and test
#   5  e2e chain on those features (same tree, network and jobs as the frozen run)

set -euo pipefail

stage=0
stop_stage=100
arm=lora32
seed=0
store=
ssl_model=utter-project/mHuBERT-147
layer=9
valid_sessions="SWH-05-20110327 SWH-15-20110310"
arpa=data/local/swahili.arpa
unigrams=asr_swahili/LM/swahili.vocab
feats_nj=8

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

case $arm in
  lora32) train_opts="--lora-rank 32 --lora-ffn --train-layer-norms --curriculum-epochs 2 --lr 1e-4" ;;
  full)   train_opts="--curriculum-epochs 2 --lr 5e-5" ;;
  *) echo "$0: --arm must be lora32 or full, got '$arm'"; exit 1 ;;
esac
sfx=$([ $seed = 0 ] || echo _s$seed)
out=exp_ft/mhubert147_$arm$sfx
tag=ft_$arm$sfx

run() { [ $stage -le $1 ] && [ $stop_stage -ge $1 ]; }

if run 0; then
  printf '%s\n' $valid_sessions > data/valid_sessions.txt
  for x in train valid; do
    d=data/${x}_ctc; rm -rf $d; mkdir -p $d
    if [ $x = valid ]; then
      awk 'NR==FNR{v[$1]=1;next} ($2 in v){print $1}' data/valid_sessions.txt data/train/utt2spk > $d/utts
    else
      awk 'NR==FNR{v[$1]=1;next} !($2 in v){print $1}' data/valid_sessions.txt data/train/utt2spk > $d/utts
    fi
    utils/filter_scp.pl $d/utts data/train/text | \
      awk '{printf "%s", $1; for (i = 2; i <= NF; i++) if ($i !~ /^</) {
              w = $i; gsub(/[^a-z0-9'"'"'-]/, "", w); if (w != "") printf " %s", w }
            print ""}' | awk 'NF > 1' > $d/text
    utils/filter_scp.pl $d/text data/train/wav.scp > $d/wav.scp
    utils/filter_scp.pl $d/text data/train/utt2spk > $d/utt2spk
    utils/utt2spk_to_spk2utt.pl $d/utt2spk > $d/spk2utt
    rm $d/utts
    utils/validate_data_dir.sh --no-feats $d
  done
fi

if run 1; then
  python local/finetune/finetune_ctc.py --ssl-model $ssl_model $train_opts --seed $seed \
    --epochs 40 --batch-utts 2 --grad-accum 1 --warmup-frac 0.1 --hold-frac 0.3 \
    --train-dir data/train_ctc --valid-dir data/valid_ctc --out-dir $out
fi

if run 2; then
  for x in valid_ctc test; do
    python local/finetune/score_ctc.py --ctc-dir $out/ctc --data-dir data/$x
  done
fi

if run 3; then
  log=$out/lm_valid.log
  python local/finetune/score_ctc_lm.py --ctc-dir $out/ctc --data-dir data/valid_ctc \
    --arpa $arpa --unigrams $unigrams --tune-utts 100000 | tee $log
  read -r alpha beta < <(sed -n 's/^best on tune set: alpha=\([^ ]*\) beta=\(.*\)$/\1 \2/p' $log)
  python local/finetune/score_ctc_lm.py --ctc-dir $out/ctc --data-dir data/test \
    --arpa $arpa --unigrams $unigrams --alphas $alpha --betas $beta | tee $out/lm_test.log
fi

if run 4; then
  for x in train_spe2e test; do
    d=data/${x}_$tag; rm -rf $d
    if [ -n "$store" ]; then
      rm -rf $store/${x}_$tag; mkdir -p $store/${x}_$tag; ln -sfn $store/${x}_$tag $d
    fi
    utils/copy_data_dir.sh data/$x $d
    shared/make_ssl.sh --cmd "$train_cmd" --nj $feats_nj --ssl-model $out --layer $layer $d
    steps/compute_cmvn_stats.sh $d
    utils/fix_data_dir.sh $d
  done
  cp data/train_spe2e/allowed_lengths.txt data/train_spe2e_$tag/
fi

if run 5; then
  local/chain/e2e/run_tdnn_flatstart.sh --affix n10_$tag \
    --input-style renorm-linear --num-tdnnf-layers 10 \
    --num-jobs-initial 1 --num-jobs-final 1 \
    --train-set train_spe2e_$tag --test-set test_$tag --stage 2
  echo "$0: chain on fine-tuned features $(grep -h WER exp/chain/e2e_tdnnf_n10_$tag/decode*/wer_* | utils/best_wer.sh)"
fi
