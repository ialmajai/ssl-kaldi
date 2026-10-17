#!/usr/bin/env bash
# Re-decode the paper's GRID systems on the full 4,000-clip unseen-speaker test set
# (the published 3,985 came from a stale pruning) and on Lombard GRID resampled to
# 25 fps (published Lombard runs fed ~24 fps video to AV-HuBERT). No retraining:
# features are re-extracted for each system's encoder, then decoded with the
# existing models. New decodes go to decode_r_test / decode_r_lomb; published
# decode dirs are left untouched. Restartable: finished steps are skipped.
#   ./run_rescore_r4000_lomb25.sh            # everything
#   STAGES="0 1" ./run_rescore_r4000_lomb25.sh
# Stage 4's test column is the best LM setting on test; the paper's GRID figures
# use the setting chosen on the Lombard dev speakers (local/score_lombard_pooled.sh).
set -euo pipefail
cd "$(dirname "$0")"
. ./cmd.sh; . ./path.sh
STAGES=${STAGES:-"0 1 2 3 4"}
SETS=${SETS:-"test lomb"}   # SETS=test skips the 25 fps Lombard set
step() { echo "######## $(date +%H:%M:%S) $*"; }
want() { [[ " $STAGES " == *" $1 "* ]]; }

# tag ckpt layer
CONFIGS="l4:input/base_vox_iter5.pt:4
l13:input/large_vox_iter5.pt:13
lor10:input/avhubert_base_ctc_lora.pt:10
ful10:input/avhubert_base_ctc_full.pt:10
lgl20:input/avhubert_large_ctc_lora.pt:20
lgl24:input/avhubert_large_ctc_lora.pt:24
lgf20:input/avhubert_large_ctc_full.pt:20
frzbase8:input/base_vox_iter5.pt:8
frzlarge16:input/large_vox_iter5.pt:16
lorabase10a025:input/avhubert_base_ctc_lora_a025.pt:10
fullbase10a025:input/avhubert_base_ctc_full_a025.pt:10
frzbase6:input/base_vox_iter5.pt:6
frzbase10:input/base_vox_iter5.pt:10
frzlarge20:input/large_vox_iter5.pt:20"
CONFIGS="$CONFIGS ${EXTRA_CONFIGS:-}"   # tag:ckpt:layer entries added by queue scripts

# chain tag graph
E2E_CHAINS="e2e_tdnnf_l4norelu:l4:exp/chain/e2e_tree/graph
e2e_tdnnf_l4norelu_cmvn:l4:exp/chain/e2e_tree/graph
e2e_tdnnf_l13norelu:l13:exp/chain/e2e_tree/graph
e2e_tdnnf_l13norelu_cmvn:l13:exp/chain/e2e_tree/graph
e2e_tdnnf_lor10norelu_cmvn:lor10:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_ful10norelu_cmvn:ful10:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_lgl20norelu_cmvn:lgl20:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_lgl24norelu_cmvn:lgl24:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_lgf20norelu_cmvn:lgf20:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_lor10norelu_cmvn_aug2x:lor10:exp/chain/e2e_tree_lor10_aug2x/graph
e2e_tdnnf_frzbase8norelu_cmvn:frzbase8:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_frzlarge16norelu_cmvn:frzlarge16:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_lorabase10a025norelu_cmvn:lorabase10a025:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_fullbase10a025norelu_cmvn:fullbase10a025:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_frzbase6norelu_cmvn:frzbase6:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_frzbase10norelu_cmvn:frzbase10:exp/chain/e2e_tree_ft/graph
e2e_tdnnf_frzlarge20norelu_cmvn:frzlarge20:exp/chain/e2e_tree_ft/graph"
# EXTRA_CHAINS="chain:tag:graph ..." adds repeat-seed chains without editing this list
E2E_CHAINS="$E2E_CHAINS ${EXTRA_CHAINS:-}"

# tag gmm-exp pca-model fmllr-chain
GMM_SYSTEMS="l4:exp-sweep-mf50-rt/l4:layer_sweep_mf/pca-30d-l4.pt:tdnntdnnf_l4
l13:exp-sweep-mfL50-rt/l13:layer_sweep_mfL/pca-30d-L13-recovered.pt:tdnntdnnf_l13"

nj_for() { local n; n=$(wc -l < $1/spk2utt); echo $(( n < 8 ? n : 8 )); }

if want 0; then
  step "stage 0: full test list (4,000) and Lombard list (2,580)"
  if [ ! -s data/r_test/text ]; then
    # grid_data_prep.sh writes data/{train,test} in its cwd; run it aside so the
    # recipe's own stage-0 dirs are not touched.
    w=$(mktemp -d); ln -s $PWD/{path.sh,utils,steps,local,input} $w/
    (cd $w && local/grid_data_prep.sh $(readlink -f $OLDPWD/grid-corpus) >/dev/null)
    rm -rf data/r_test; cp -r $w/data/test data/r_test; rm -rf $w
  fi
  [ -s data/r_lomb/text ] || python local/lombard_data_prep.py \
    --video-dir lombard-grid/lombardgrid/front --out data/r_lomb
  if [ ! -f input/avhubert_large_ctc_full.pt ]; then
    step "re-export the large full fine-tune encoder"
    PYTHONPATH=$PWD/av_hubert/fairseq:$PWD/av_hubert AVHUBERT_ROOT=$PWD/av_hubert \
      python local/finetune/export_lora_encoder.py \
      finetune_lora/exp_large_ctc_full/checkpoints/checkpoint_best.pt \
      input/large_vox_iter5.pt input/avhubert_large_ctc_full.pt
  fi
  echo "test: $(wc -l < data/r_test/text) utts, lombard: $(wc -l < data/r_lomb/text) utts"
fi

if want 1; then
  for c in $CONFIGS; do
    IFS=: read tag ckpt layer <<< "$c"
    [ -n "${ONLY_TAGS:-}" ] && [[ " $ONLY_TAGS " != *" $tag "* ]] && continue
    for set in $SETS; do
      d=data/r_${set}_${tag}_raw
      if [ ! -s $d/feats.scp ]; then
        step "stage 1: extract $tag (L$layer) on $set"
        min=$([ $set = lomb ] && echo 30 || echo 74)
        rm -rf $d; local/copy_data_dir.sh data/r_$set $d
        local/make_avhubert.sh --cmd "$feats_cmd" --nj 2 --ckpt $ckpt \
          --avhubert-path $PWD/av_hubert --layer $layer --crop-mode meanface \
          --min-frames $min $d
        utils/fix_data_dir.sh $d
      fi
      if [ ! -s ${d}_upsampled/feats.scp ]; then
        rm -rf ${d}_upsampled; local/copy_data_dir.sh $d ${d}_upsampled
        shared/upsample_features.sh --cmd "$decode_cmd" --nj 4 --interp-mode nearest \
          --upsample-factor 2 $d ${d}_upsampled
        steps/compute_cmvn_stats.sh ${d}_upsampled; utils/fix_data_dir.sh ${d}_upsampled
      fi
      echo "$d: $(wc -l < $d/feats.scp) utts"
    done
  done
fi

if want 2; then
  for c in $E2E_CHAINS; do
    IFS=: read chain tag graph <<< "$c"
    for set in $SETS; do
      out=exp/chain/$chain/decode_r_$set; data=data/r_${set}_${tag}_raw_upsampled
      [ -f $out/.done ] && continue
      [ -f exp/chain/$chain/final.mdl ] || { step "skip $chain: not trained yet"; continue; }
      step "stage 2: decode $chain on $set"
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --lattice-beam 2.0 \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk 150 --nj $(nj_for $data) --num-threads 2 --cmd "$decode_cmd" \
        $graph $data $out
      touch $out/.done
    done
  done
fi

if want 3; then
  for s in $GMM_SYSTEMS; do
    IFS=: read tag E pca fchain <<< "$s"
    for set in $SETS; do
      raw=data/r_${set}_${tag}_raw; p=data/r_${set}_${tag}_pca
      if [ ! -s ${p}_up/feats.scp ]; then
        step "stage 3: PCA + upsample $tag $set"
        rm -rf $p ${p}_up; local/copy_data_dir.sh $raw $p
        shared/make_pca_features.sh --cmd "$decode_cmd" --nj 4 --pca-model $pca $raw $p
        steps/compute_cmvn_stats.sh $p; utils/fix_data_dir.sh $p
        local/copy_data_dir.sh $p ${p}_up
        shared/upsample_features.sh --cmd "$decode_cmd" --nj 4 --interp-mode nearest \
          --upsample-factor 2 $p ${p}_up
        steps/compute_cmvn_stats.sh ${p}_up; utils/fix_data_dir.sh ${p}_up
      fi
      nj=$(nj_for ${p}_up)
      for m in mono tri1 tri2b; do
        [ -f $E/$m/decode_r_$set/.done ] && continue
        step "stage 3: decode $E/$m on $set"
        steps/decode.sh --nj $nj --cmd "$decode_cmd" $E/$m/graph ${p}_up $E/$m/decode_r_$set
        touch $E/$m/decode_r_$set/.done
      done
      if [ ! -f $E/tri3b/decode_r_$set/.done ]; then
        step "stage 3: decode_fmllr $E/tri3b on $set"
        steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" $E/tri3b/graph ${p}_up $E/tri3b/decode_r_$set
        touch $E/tri3b/decode_r_$set/.done
      fi
      f=data-fmllr-tri3b/r_${set}_${tag}_up
      if [ ! -s $f/feats.scp ]; then
        rm -rf $f
        steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
          --transform-dir $E/tri3b/decode_r_$set $f ${p}_up $E/tri3b $f/log $f/data
        steps/compute_cmvn_stats.sh $f $f/log $f/data
      fi
      out=exp/chain/$fchain/decode_r_$set
      if [ ! -f $out/.done ]; then
        step "stage 3: decode $fchain on $set"
        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj $nj --cmd "$decode_cmd" \
          exp/chain/$fchain/graph $f $out
        touch $out/.done
      fi
    done
  done
fi

best() { grep -h WER $1/wer_* 2>/dev/null | sort -k2 -g | awk 'NR==1{print $2, $4}'; }
old_subset() {  # WER of a decode restricted to the published 3,985 test utterances
  local d=$1 best_f lw lm wip h
  best_f=$(grep -H WER $d/wer_* | sort -t' ' -k2 -g | awk -F: 'NR==1{print $1}')
  lw=$(basename $best_f | sed 's/^wer_//'); lm=${lw%_*}; wip=${lw#*_}
  h=$d/scoring_kaldi/penalty_$wip/$lm.txt
  utils/filter_scp.pl data/test_l4_raw/feats.scp $h \
    | compute-wer --text --mode=present ark:data/r_test/text ark:- 2>/dev/null | awk '/WER/{print $2}'
}

if want 4; then
  step "stage 4: results"
  printf "%-36s %-16s %-10s %-22s %-18s\n" system "test (WER words)" "old-3985" "lombard full" "lombard dev/eval"
  for c in $E2E_CHAINS; do
    IFS=: read chain tag graph <<< "$c"; d=exp/chain/$chain
    printf "%-36s %-16s %-10s %-22s %-18s\n" $chain "$(best $d/decode_r_test)" "$(old_subset $d/decode_r_test)" \
      "$(best $d/decode_r_lomb)" "$(local/score_lombard_split.sh $d/decode_r_lomb data/lang/words.txt data/r_lomb_${tag}_raw 2>/dev/null)"
  done
  for s in $GMM_SYSTEMS; do
    IFS=: read tag E pca fchain <<< "$s"
    for d in $E/mono $E/tri1 $E/tri2b $E/tri3b exp/chain/$fchain; do
      printf "%-36s %-16s %-10s %-22s %-18s\n" "$d" "$(best $d/decode_r_test)" "$(old_subset $d/decode_r_test)" \
        "$(best $d/decode_r_lomb)" "$(local/score_lombard_split.sh $d/decode_r_lomb data/lang/words.txt data/r_lomb_${tag}_raw 2>/dev/null)"
    done
  done
fi
echo "RESCORE-DONE"
