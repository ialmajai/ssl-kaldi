#!/usr/bin/env bash
# Frame-rate control for the icefall comparison (2026-10-16): the large LoRA L20
# e2e chain with CMVN off, trained on the raw 25 fps features with frame
# subsampling 1, against e2e_tdnnf_lgl20norelu (same encoder, recipe and seed,
# but x2-upsampled 50 fps input with frame subsampling 2). Both chains output
# at 25 fps, so only the input upsampling differs.
set -uo pipefail
. ./cmd.sh; . ./path.sh
affix=lgl20norelu_25fps; dir=exp/chain/e2e_tdnnf_$affix; graph=exp/chain/e2e_tree_ft/graph

# GRID clips are 3 s: 75 frames at 25 fps (74 for 61 clips).
echo -e '2.96\n3.00' > data/train_lgl20_raw/allowed_durs.txt
echo -e '74\n75' > data/train_lgl20_raw/allowed_lengths.txt

if [ ! -f $dir/final.mdl ]; then
  echo "$(date +%H:%M) training $dir"
  local/chain/e2e/run_tdnn_flatstart.sh --affix $affix --srand 0 --frame-subsampling-factor 1 \
    --feat-dim 1024 --cmvn-opts "--norm-means=false --norm-vars=false" \
    --treedir exp/chain/e2e_tree_ft --train-set train_lgl20_raw --test-set r_test_lgl20_raw \
    --stage 2 > logs_25fps_train.log 2>&1 || { echo "FAILED training"; exit 1; }
fi

for set in lomb_l lomb_p; do
  out=$dir/decode_s_$set; data=data/s_${set}_lgl20_raw
  [ -f $out/.done ] && continue
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --lattice-beam 2.0 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --frames-per-chunk 150 --nj 8 --num-threads 2 --cmd "$decode_cmd" \
    $graph $data $out >> logs_25fps_decode.log 2>&1 || { echo "FAILED decode $set"; exit 1; }
  touch $out/.done
done
echo "$(date +%H:%M) Lombard speech: $(local/score_lombard_split.sh $dir/decode_s_lomb_l data/lang/words.txt data/s_lomb_l_lgl20_raw)"
echo "$(date +%H:%M) plain speech:   $(local/score_lombard_split.sh $dir/decode_s_lomb_p data/lang/words.txt data/s_lomb_p_lgl20_raw)"
pooled=$(local/score_lombard_pooled.sh $dir/decode_s_lomb_l $dir/decode_s_lomb_p)
echo "$(date +%H:%M) pooled 5,142:   $pooled"
# Score GRID at the LM setting chosen on the Lombard dev speakers.
echo "$(date +%H:%M) GRID test:      $(grep WER $dir/decode/wer_${pooled##*setting=})"
echo FPS25-DONE
