#!/usr/bin/env bash
# Decode every paper GRID system on the Lombard GRID clip set shared with the
# companion icefall paper (2026-10-12): local/lombard_clips.txt, 5,142 front-view
# clips (2,580 Lombard + 2,562 plain; under 2 s or undecodable excluded), mean-face
# crops from the shared .mfroi.npz caches (MediaPipe fallback, so no tracking
# losses), native ~24 fps. Data dirs data/s_lomb_{l,p}; decodes go to
# decode_s_lomb_{l,p}. Scoring: LM weight chosen on the 14 dev speakers, reported
# on the 40 eval speakers and on all 54. Restartable; systems whose model is
# missing are skipped.
#   ./run_shared_lombard.sh            # everything
#   STAGES="4" ./run_shared_lombard.sh # table only
set -euo pipefail
cd "$(dirname "$0")"
. ./cmd.sh; . ./path.sh
STAGES=${STAGES:-"1 2 3 4"}
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

if want 1; then
  for c in $CONFIGS; do
    IFS=: read tag ckpt layer <<< "$c"
    [ -n "${ONLY_TAGS:-}" ] && [[ " $ONLY_TAGS " != *" $tag "* ]] && continue
    for set in lomb_l lomb_p; do
      d=data/s_${set}_${tag}_raw
      if [ ! -s $d/feats.scp ]; then
        step "stage 1: extract $tag (L$layer) on $set"
        min=30
        rm -rf $d; local/copy_data_dir.sh data/s_$set $d
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
    [ -n "${ONLY_TAGS:-}" ] && [[ " $ONLY_TAGS " != *" $tag "* ]] && continue
    for set in lomb_l lomb_p; do
      out=exp/chain/$chain/decode_s_$set; data=data/s_${set}_${tag}_raw_upsampled
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
    for set in lomb_l lomb_p; do
      raw=data/s_${set}_${tag}_raw; p=data/s_${set}_${tag}_pca
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
        [ -f $E/$m/decode_s_$set/.done ] && continue
        step "stage 3: decode $E/$m on $set"
        steps/decode.sh --nj $nj --cmd "$decode_cmd" $E/$m/graph ${p}_up $E/$m/decode_s_$set
        touch $E/$m/decode_s_$set/.done
      done
      if [ ! -f $E/tri3b/decode_s_$set/.done ]; then
        step "stage 3: decode_fmllr $E/tri3b on $set"
        steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" $E/tri3b/graph ${p}_up $E/tri3b/decode_s_$set
        touch $E/tri3b/decode_s_$set/.done
      fi
      f=data-fmllr-tri3b/r_${set}_${tag}_up
      if [ ! -s $f/feats.scp ]; then
        rm -rf $f
        steps/nnet/make_fmllr_feats.sh --nj $nj --cmd "$train_cmd" \
          --transform-dir $E/tri3b/decode_s_$set $f ${p}_up $E/tri3b $f/log $f/data
        steps/compute_cmvn_stats.sh $f $f/log $f/data
      fi
      out=exp/chain/$fchain/decode_s_$set
      if [ ! -f $out/.done ]; then
        step "stage 3: decode $fchain on $set"
        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --nj $nj --cmd "$decode_cmd" \
          exp/chain/$fchain/graph $f $out
        touch $out/.done
      fi
    done
  done
fi

# Each cell: dev / eval-40 / all-54, at the LM weight and penalty best on dev.
split() { local_score=$(local/score_lombard_split.sh $1 data/lang/words.txt $2 2>/dev/null) || return 0
  echo "$local_score" | sed -E 's/dev=([^ ]*) eval=([^ ]*) all=([^ ]*).*/\1 \/ \2 \/ \3/'; }

if want 4; then
  step "stage 4: results (dev / eval-40 / all-54)"
  printf "%-40s %-24s %-24s\n" system "Lombard speech" "plain speech"
  for c in $E2E_CHAINS; do
    IFS=: read chain tag graph <<< "$c"; d=exp/chain/$chain
    [ -f $d/decode_s_lomb_l/.done ] || continue
    printf "%-40s %-24s %-24s\n" $chain "$(split $d/decode_s_lomb_l data/s_lomb_l_${tag}_raw)" \
      "$(split $d/decode_s_lomb_p data/s_lomb_p_${tag}_raw)"
  done
  for s in $GMM_SYSTEMS; do
    IFS=: read tag E pca fchain <<< "$s"
    for d in $E/mono $E/tri1 $E/tri2b $E/tri3b exp/chain/$fchain; do
      printf "%-40s %-24s %-24s\n" "$d" "$(split $d/decode_s_lomb_l data/s_lomb_l_${tag}_raw)" \
        "$(split $d/decode_s_lomb_p data/s_lomb_p_${tag}_raw)"
    done
  done
fi
echo "SHARED-LOMBARD-DONE"
